from urllib.parse import urlencode
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ceda-tools-search")
client = httpx.Client(timeout=30.0)


@mcp.tool()
def search_observations(search: str, limit: int = 25, offset: int = 0) -> dict:
    """
    Text search catalogue observations.
    Returns:
      dict with {count, results, ...} from v3 API
    """
    params = {"search": search, "limit": limit, "offset": offset}
    url = f"https://catalogue.ceda.ac.uk/api/v3/observations/?{urlencode(params)}"
    r = client.get(url)
    r.raise_for_status()
    return r.json()

if __name__ == "__main__":
    # Run as an MCP server over stdio locally
    mcp.run()

