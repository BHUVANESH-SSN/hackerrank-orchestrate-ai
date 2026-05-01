"""
Async Retrieval Blueprint.

Demonstrates drastically lowering Latency by executing multiple Tool Calls (like checking SQL DB,
Checking Vector DB, and Checking Salesforce) CONCURRENTLY rather than sequentially.
"""
import asyncio
import time

async def query_chroma_db(query: str) -> str:
    """Simulate a vector database query (~800ms)"""
    await asyncio.sleep(0.8)
    return "VectorDB: Policy allows refunds within 30 days."

async def query_user_sql_db(ticket_id: str) -> str:
    """Simulate a SQL look up for user metadata (~500ms)"""
    await asyncio.sleep(0.5)
    return f"SQL: User associated with ticket {ticket_id} is on the Premium Plan."

async def run_parallel_retrieval(query: str, ticket_id: str):
    print("Fetching tools sequentially would take 1.3 seconds...")
    start_time = time.time()
    
    # Run both exact queries natively in parallel!
    results = await asyncio.gather(
        query_chroma_db(query),
        query_user_sql_db(ticket_id)
    )
    
    end_time = time.time()
    print(f"Parallel Execution completed in {end_time - start_time:.2f} seconds!")
    print("\nSynthesizing Context:")
    print(f"- {results[0]}")
    print(f"- {results[1]}")
    
    return results

if __name__ == "__main__":
    asyncio.run(run_parallel_retrieval("I want a refund", "TCK-123"))
