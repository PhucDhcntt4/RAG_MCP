
import json

import httpx
import pytest

from rag_mcp_client.config import Settings
from rag_mcp_client.models import SearchRequest
from rag_mcp_client.rag_client import (
    RagClientError,
    get_knowledge_asset,
    get_product_detail,
    get_product_image,
    search_knowledge,
    search_products,
)


def create_test_settings() -> Settings:
    return Settings(
        search_api_key="test-key",
        rag_search_url="http://rag.test/api/v1/knowledge/search",
        rag_products_url="http://rag.test/api/v1/products",
    )


@pytest.mark.asyncio
async def test_search_knowledge_success() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == httpx.URL(
            "http://rag.test/api/v1/knowledge/search"
        )
        assert request.headers["Authorization"] == "Bearer test-key"

        request_body = json.loads(request.content)
        assert request_body == {
            "query": "Chính sách bảo hành là gì?",
            "top_k": 3,
        }

        return httpx.Response(
            status_code=200,
            json={
                "success": True,
                "status": "knowledge_found",
                "content": "Nội dung chính sách bảo hành.",
                "sources": [
                    {
                        "document_id": 123,
                        "source_key": "warranty-policy",
                        "title": "Chính sách bảo hành",
                        "category": "warranty",
                        "chunk_index": 0,
                        "similarity": 0.91,
                    }
                ],
                "elapsed_ms": 120.5,
            },
        )

    result = await search_knowledge(
        SearchRequest(
            query="Chính sách bảo hành là gì?",
            top_k=3,
        ),
        settings=create_test_settings(),
        transport=httpx.MockTransport(handler),
    )

    assert result.success is True
    assert result.status == "knowledge_found"
    assert result.elapsed_ms == 120.5
    assert len(result.sources) == 1
    assert result.sources[0].category == "warranty"


@pytest.mark.asyncio
async def test_search_knowledge_unauthorized() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=401,
            json={"detail": "Unauthorized"},
        )

    with pytest.raises(RagClientError, match="HTTP 401"):
        await search_knowledge(
            SearchRequest(query="test"),
            settings=create_test_settings(),
            transport=httpx.MockTransport(handler),
        )


@pytest.mark.asyncio
async def test_get_knowledge_asset_success() -> None:
    image_bytes = b"\x89PNG\r\n\x1a\nimage-data"

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url == httpx.URL(
            "http://rag.test/api/v1/knowledge/assets/2"
        )
        assert request.headers["Authorization"] == "Bearer test-key"
        assert request.headers["Accept"] == "image/*"

        return httpx.Response(
            status_code=200,
            content=image_bytes,
            headers={"Content-Type": "image/png"},
        )

    data, media_type = await get_knowledge_asset(
        2,
        settings=create_test_settings(),
        transport=httpx.MockTransport(handler),
    )

    assert data == image_bytes
    assert media_type == "image/png"


@pytest.mark.asyncio
async def test_get_knowledge_asset_rejects_non_image() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={"detail": "not an image"},
        )

    with pytest.raises(RagClientError, match="không phải là hình ảnh"):
        await get_knowledge_asset(
            2,
            settings=create_test_settings(),
            transport=httpx.MockTransport(handler),
        )


@pytest.mark.asyncio
async def test_search_products_success() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/v1/products"
        assert request.url.params["q"] == "S5921"
        assert request.url.params["limit"] == "3"
        assert request.headers["Authorization"] == "Bearer test-key"

        return httpx.Response(
            status_code=200,
            json={
                "products": [
                    {
                        "point_id": "product-point",
                        "product_code": "S5921",
                        "title": "Sandal Kẹp Nữ Đông Hải",
                        "image_url": "https://cdn.example/product.jpg",
                    }
                ],
                "next_cursor": None,
                "limit": 3,
                "query": "S5921",
                "product_type": "",
                "status": "",
            },
        )

    result = await search_products(
        "S5921",
        limit=3,
        settings=create_test_settings(),
        transport=httpx.MockTransport(handler),
    )

    assert len(result.products) == 1
    assert result.products[0].product_code == "S5921"


@pytest.mark.asyncio
async def test_get_product_detail_success() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/v1/products/S5921"
        assert request.headers["Authorization"] == "Bearer test-key"

        return httpx.Response(
            status_code=200,
            json={
                "point_id": "product-point",
                "product": {
                    "product_code": "S5921",
                    "title": "Sandal Kẹp Nữ Đông Hải",
                    "public_info": {
                        "product_code": "S5921",
                        "product_name": "Sandal Kẹp Nữ Đông Hải",
                        "prices": [620000],
                        "colors": ["Kem"],
                        "available_sizes": ["37"],
                        "image_urls": [
                            "https://cdn.example/product.jpg"
                        ],
                    },
                    "variants": [
                        {
                            "color": "Kem",
                            "size": "37",
                            "price": 620000,
                            "inventory_quantity": 11,
                            "available": True,
                        }
                    ],
                },
            },
        )

    result = await get_product_detail(
        "S5921",
        settings=create_test_settings(),
        transport=httpx.MockTransport(handler),
    )

    assert result.product.product_code == "S5921"
    assert result.product.variants[0].inventory_quantity == 11


@pytest.mark.asyncio
async def test_get_product_image_success() -> None:
    image_bytes = b"\xff\xd8\xffproduct-image"

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "cdn.shopify.com"
        assert request.url.params["width"] == "600"
        assert request.headers["Accept"] == "image/*"
        return httpx.Response(
            status_code=200,
            content=image_bytes,
            headers={"Content-Type": "image/jpeg"},
        )

    data, media_type = await get_product_image(
        "https://cdn.shopify.com/product.jpg",
        transport=httpx.MockTransport(handler),
    )

    assert data == image_bytes
    assert media_type == "image/jpeg"


@pytest.mark.asyncio
async def test_get_product_image_rejects_untrusted_host() -> None:
    with pytest.raises(RagClientError, match="không được phép"):
        await get_product_image("https://example.com/product.jpg")
