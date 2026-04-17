"""
Demo 8 – Resumable AI Procurement Agent (LangGraph Persistence + Interrupt)

Scenario: An AI agent handles purchase requests. When a purchase exceeds
€10,000 it must pause for manager approval — which may come hours or days later.

The graph:

  START → lookup_vendors → fetch_pricing → compare_quotes
        → request_approval (INTERRUPTS here — process exits!)
        → submit_purchase_order → notify_employee → END

To simulate a real-world "late second invocation" across process restarts,
we use SqliteSaver (file-based checkpoint) and two CLI modes:

  python demo8.1-purchase-agent.py              # First run  — steps 1-3, then suspends
  python demo8.1-purchase-agent.py --resume     # Second run — manager approves, steps 5-6

Between the two runs the Python process exits completely.  The full agent
state (vendor data, pricing, chosen quote) survives on disk in SQLite.
"""

import sys
import os
import sqlite3
import time
from typing import Annotated, TypedDict
import requests
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

# ─── State ────────────────────────────────────────────────────────────────────

class ProcurementState(TypedDict):
    request: str
    quantity: int
    vendors: list[dict]
    quotes: list[dict]
    best_quote: dict
    approval_status: str
    po_number: str
    notification: str


# ─── LLM (used only for the notification step to make it feel "agentic") ─────

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key="your_google_api_key") # Replace with your google_api_key

# ─── Tool ───────────────────────────────────────────────────────
@tool
def get_unit_price(product_name: str) -> float:
    """Fetches the live unit price for a laptop from the catalog."""
    url = "https://dummyjson.com/products/category/laptops"
    resp = requests.get(url)
    if resp.status_code == 200:
        products = resp.json().get("products", [])
        for p in products:
            if product_name.lower() in p["title"].lower():
                return float(p["price"])
    return 999.0 

class QuantityExtractor(BaseModel):
    quantity: int = Field(description="The number of units requested")

# ─── Node functions ──────────────────────────────────────────────────────────

def lookup_vendors(state: ProcurementState) -> dict:
    """Step 1: Extract quantity and fetch live vendors from API"""
    print("\n[Step 1] Parsing request and fetching live vendors...")
    
    extractor = llm.with_structured_output(QuantityExtractor)
    res = extractor.invoke(state["request"])
    qty = res.quantity

    resp = requests.get("https://dummyjson.com/products/category/laptops")
    all_prods = resp.json().get("products", [])
    fast_prods = [p for p in all_prods if "month" not in p.get("shippingInformation", "").lower()]
    fast_prods.sort(key=lambda x: x["price"])
    
    vendors = [{"name": p["title"], "delivery": p["shippingInformation"]} for p in fast_prods[:3]]
    for v in vendors:
        print(f"   Found: {v['name']} ({v['delivery']})")
        
    return {"vendors": vendors, "quantity": qty}


def fetch_pricing(state: ProcurementState) -> dict:
    """Step 2: Use tool calling to get prices"""
    print("\n[Step 2] Fetching pricing via tool calls...")
    quotes = []
    qty = state.get("quantity", 1)
    
    for v in state["vendors"]:
        price = get_unit_price.invoke({"product_name": v["name"]})
        total = price * qty
        
        print(f"   {v['name']}: €{price}/unit x {qty} = €{total:,}")
        
        quotes.append({
            "vendor": v["name"], 
            "unit_price": price, 
            "total": total, 
            "delivery_days": v["delivery"]
        })
    return {"quotes": quotes}


def compare_quotes(state: ProcurementState) -> dict:
    """Step 3: Compare quotes and pick the best one."""
    print("\n[Step 3] Comparing quotes...")
    time.sleep(0.5)
    best = min(state["quotes"], key=lambda q: q["total"])
    print(f"   Best quote: {best['vendor']} at €{best['total']:,}")
    print(f"   (Saves €{max(q['total'] for q in state['quotes']) - best['total']:,} "
          f"vs most expensive option)")
    return {"best_quote": best}


def request_approval(state: ProcurementState) -> dict:
    """Step 4: Human-in-the-loop — request manager approval for orders > €10,000."""
    best = state["best_quote"]
    qty = state["quantity"]
    print("\n[Step 4] Order exceeds €10,000 — manager approval required!")
    print(f"   Sending approval request to manager...")
    amount_str = f"€{best['total']:,}"
    delivery_str = f"{best['delivery_days']}"
    print(f"   ┌─────────────────────────────────────────────┐")
    print(f"   │  APPROVAL NEEDED                            │")
    print(f"   │  Vendor:   {best['vendor']:<33}│")
    print(f"   │  Amount:   {amount_str:<33}│")
    print(f"   │  Items:    {qty} laptops for engineering team  │")
    print(f"   │  Delivery: {delivery_str:<33}│")
    print(f"   └─────────────────────────────────────────────┘")

    # ── THIS IS WHERE THE MAGIC HAPPENS ──
    # interrupt() freezes the entire graph state into the checkpoint store.
    # The process can now exit completely. When resumed later (even days later),
    # execution continues right here with the resume value.
    decision = interrupt({
        "message": f"Approve purchase of {qty} laptops from {best['vendor']} for €{best['total']:,}?",
        "vendor": best["vendor"],
        "amount": best["total"],
    })

    print(f"\n[Step 4] Manager responded: {decision}")
    return {"approval_status": decision}


def submit_purchase_order(state: ProcurementState) -> dict:
    """Step 5: Submit the purchase order to the ERP system."""
    approval = state.get("approval_status", "").lower()

    if "reject" in approval:
        print("\n[Step 5] Purchase REJECTED by manager. Aborting.")
        return {"po_number": "REJECTED"}

    print("\n[Step 5] Submitting purchase order to ERP system...")
    time.sleep(1)
    po_number = "PO-2026-00342"
    print(f"   Purchase order created: {po_number}")
    print(f"   Vendor: {state['best_quote']['vendor']}")
    print(f"   Amount: €{state['best_quote']['total']:,}")
    return {"po_number": po_number}


def notify_employee(state: ProcurementState) -> dict:
    """Step 6: Handle both Approval and Rejection context (Task 3)."""
    print("\n[Step 6] Notifying employee...")

    raw_status = state.get("approval_status", "Auto-approved (Under €10,000 threshold)")
    is_rejected = "reject" in raw_status.lower()
    
    context = f"Status: {'Rejected' if is_rejected else 'Approved'}. Note: {raw_status}"
    prompt = f"Draft a short, professional email to an employee about their order of {state['quantity']} laptops. Context: {context}"
    
    response = llm.invoke(prompt)
    print(f"   Notification: \"{response.content}\"")
    return {"notification": response.content}

# ─── Routing Functions  ─────────────────────────────────────────
def route_approval(state: ProcurementState):
    if state["best_quote"]["total"] > 10000:
        return "request_approval"
    return "submit_purchase_order"

def route_rejection(state: ProcurementState):
    if "reject" in state["approval_status"].lower():
        return "notify_employee"
    return "submit_purchase_order"


# ─── Build the graph ─────────────────────────────────────────────────────────
#
#   START → lookup_vendors → fetch_pricing → compare_quotes
#         → request_approval (INTERRUPT)
#         → submit_purchase_order → notify_employee → END

builder = StateGraph(ProcurementState)

builder.add_node("lookup_vendors", lookup_vendors)
builder.add_node("fetch_pricing", fetch_pricing)
builder.add_node("compare_quotes", compare_quotes)
builder.add_node("request_approval", request_approval)
builder.add_node("submit_purchase_order", submit_purchase_order)
builder.add_node("notify_employee", notify_employee)

builder.add_edge(START, "lookup_vendors")
builder.add_edge("lookup_vendors", "fetch_pricing")
builder.add_edge("fetch_pricing", "compare_quotes")
builder.add_conditional_edges("compare_quotes", route_approval)
builder.add_conditional_edges("request_approval", route_rejection)
builder.add_edge("submit_purchase_order", "notify_employee")
builder.add_edge("notify_employee", END)


# ─── Checkpointer (SQLite — survives process restarts!) ──────────────────────

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "procurement_checkpoints.db")
THREAD_ID = "procurement-thread-1"
config = {"configurable": {"thread_id": THREAD_ID}}


# ─── Main ────────────────────────────────────────────────────────────────────

def run_first_invocation(graph):
    print("=" * 60)
    print("  FIRST INVOCATION — Employee submits purchase request")
    print("=" * 60)
    
    request_str = "Order 9 laptops for the sales team" # Less than 9 (or 9) triggers automated approval, while more than 9 triggers manager approval
    print(f"\nEmployee request: \"{request_str}\"")

    result = graph.invoke(
        {"request": request_str},
        config,
    )

    snapshot = graph.get_state(config)
    
    if snapshot.next: 

        print("\n" + "=" * 60)
        print(f"AGENT SUSPENDED — waiting for human input at: {snapshot.next}")
        print("=" * 60)
        print("\n  The agent process can now exit completely.")
        print("  To resume, run with --resume")
    else:

        print("\n" + "=" * 60)
        print("WORKFLOW COMPLETE — No approval required.")
        print("=" * 60)


def run_second_invocation(graph):
    """Second run: manager approves, agent wakes up at step 5 with full context."""
    print("=" * 60)
    print("  SECOND INVOCATION — Manager approves (maybe days later!)")
    print("=" * 60)

    # Show that the state survived the process restart
    saved_state = graph.get_state(config)
    if not saved_state or not saved_state.values:
        print("\nNo saved state found! Run without --resume first.")
        return

    print("\nLoading state from checkpoint...")
    print(f"  ✓ Request: {saved_state.values.get('request', 'N/A')}")
    print(f"  ✓ Vendors found: {len(saved_state.values.get('vendors', []))}")
    print(f"  ✓ Quotes received: {len(saved_state.values.get('quotes', []))}")
    best = saved_state.values.get("best_quote", {})
    print(f"  ✓ Best quote: {best.get('vendor', 'N/A')} at €{best.get('total', 0):,}")
    print(f"\n  Steps 1-3 are NOT re-executed — their output is in the checkpoint!\n")

    # Resume with the manager's approval
    print("Manager clicks [APPROVE] ...")
    time.sleep(1)

    result = graph.invoke(
        Command(resume="Approved — go ahead with the purchase."), # Change to Rejected — over budget for rejection testing (and also change request_str laptop amount to something like 50)
        config,
    )

    print("\n" + "=" * 60)
    print("PROCUREMENT COMPLETE")
    print("=" * 60)
    print(f"\n  PO Number:    {result.get('po_number', 'N/A')}")
    print(f"  Vendor:       {result.get('best_quote', {}).get('vendor', 'N/A')}")
    print(f"  Total:        €{result.get('best_quote', {}).get('total', 0):,}")
    print(f"  Approval:     {result.get('approval_status', 'N/A')}")
    print()


if __name__ == "__main__":
    resume_mode = "--resume" in sys.argv

    # Clean start if not resuming
    if not resume_mode and os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"(Cleaned up old checkpoint DB)")

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = builder.compile(checkpointer=checkpointer)

    try:
        if resume_mode:
            run_second_invocation(graph)
        else:
            run_first_invocation(graph)
    finally:
        conn.close()
