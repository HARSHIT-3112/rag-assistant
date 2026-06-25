import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest_asyncio.fixture(scope="session")
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_upload_non_pdf_rejected(client):
    resp = await client.post(
        "/api/v1/documents/upload",
        files={"file": ("test.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 415


@pytest.mark.asyncio
async def test_list_documents(client):
    resp = await client.get("/api/v1/documents/")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
