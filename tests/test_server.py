from unittest.mock import AsyncMock

import pytest

from rag_mcp_client.models import (
    ProductDetailResponse,
    ProductSearchResponse,
    SearchResponse,
)
from rag_mcp_client import server


def test_server_instructions_are_loaded_from_prompt_file() -> None:
    assert server.SERVER_INSTRUCTIONS.startswith(
        "Phân loại yêu cầu trước khi gọi tool"
    )
    assert "show_knowledge_images" in server.SERVER_INSTRUCTIONS
    assert "Đọc toàn bộ trường content" in server.SERVER_INSTRUCTIONS
    assert "không chỉ trả lời một ý ngắn" in server.SERVER_INSTRUCTIONS
    assert "Sao chép nguyên văn tin nhắn" in server.SERVER_INSTRUCTIONS
    assert "MCP tự cố định top_k=10" in server.SERVER_INSTRUCTIONS
    assert 'không diễn giải "thời gian làm việc"' in (
        server.SERVER_INSTRUCTIONS
    )
    assert "Không dừng lại sau khi tìm thấy" in (
        server.SERVER_INSTRUCTIONS
    )
    assert "Với câu hỏi về thời gian làm việc" in (
        server.SERVER_INSTRUCTIONS
    )
    assert "### Nguồn tài liệu:" in server.SERVER_INSTRUCTIONS
    assert "Dùng search_products" in server.SERVER_INSTRUCTIONS
    assert "Tồn kho phải lấy từ get_product_detail" in (
        server.SERVER_INSTRUCTIONS
    )
    assert "get_product_detail tự lấy thông tin chi tiết" in (
        server.SERVER_INSTRUCTIONS
    )


def test_local_prompt_file_can_be_edited(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_file = tmp_path / "server_instructions.txt"
    prompt_file.write_text("Hướng dẫn mới.\n", encoding="utf-8")
    monkeypatch.setattr(
        server,
        "LOCAL_SERVER_INSTRUCTIONS_PATH",
        prompt_file,
    )

    assert server.load_server_instructions() == "Hướng dẫn mới."


@pytest.mark.asyncio
async def test_search_tool_declares_output_schema() -> None:
    tools = await server.mcp.list_tools()
    search_tool = next(
        tool for tool in tools if tool.name == "search_knowledge"
    )

    assert search_tool.output_schema is not None
    assert search_tool.output_schema["type"] == "object"
    assert "status" in search_tool.output_schema["properties"]
    assert "image_asset_ids" in search_tool.output_schema["properties"]
    assert set(search_tool.input_schema["properties"]) == {
        "user_question_verbatim"
    }
    assert search_tool.input_schema["required"] == [
        "user_question_verbatim"
    ]


@pytest.mark.asyncio
async def test_product_tools_declare_expected_schemas() -> None:
    tools = await server.mcp.list_tools()
    tools_by_name = {tool.name: tool for tool in tools}

    search_tool = tools_by_name["search_products"]
    assert set(search_tool.input_schema["properties"]) == {
        "product_query",
        "limit",
    }
    assert search_tool.input_schema["required"] == ["product_query"]
    assert "products" in search_tool.output_schema["properties"]

    detail_tool = tools_by_name["get_product_detail"]
    assert set(detail_tool.input_schema["properties"]) == {
        "product_code",
        "color",
        "size",
    }
    assert detail_tool.input_schema["required"] == ["product_code"]
    assert "variants" in detail_tool.output_schema["properties"]
    assert "images" in detail_tool.output_schema["properties"]
    assert detail_tool.meta["ui"]["resourceUri"] == (
        server.PRODUCT_GALLERY_URI
    )


@pytest.mark.asyncio
async def test_only_asset_tool_is_bound_to_image_gallery() -> None:
    tools = await server.mcp.list_tools()
    search_tool = next(
        tool for tool in tools if tool.name == "search_knowledge"
    )

    assert not search_tool.meta

    asset_tool = next(
        tool for tool in tools if tool.name == "get_knowledge_asset"
    )
    assert asset_tool.meta is not None
    assert asset_tool.meta["ui"]["resourceUri"] == (
        server.KNOWLEDGE_GALLERY_URI
    )

    gallery_tool = next(
        tool for tool in tools if tool.name == "show_knowledge_images"
    )
    assert gallery_tool.meta is not None
    assert gallery_tool.meta["ui"]["resourceUri"] == (
        server.KNOWLEDGE_GALLERY_URI
    )

    resources = await server.mcp.list_resources()
    gallery = next(
        resource
        for resource in resources
        if str(resource.uri) == server.KNOWLEDGE_GALLERY_URI
    )
    assert gallery.mime_type == "text/html;profile=mcp-app"

    legacy_gallery = next(
        resource
        for resource in resources
        if str(resource.uri) == server.LEGACY_GALLERY_URI
    )
    assert legacy_gallery.mime_type == "text/html+skybridge"

    product_gallery = next(
        resource
        for resource in resources
        if str(resource.uri) == server.PRODUCT_GALLERY_URI
    )
    assert product_gallery.mime_type == "text/html;profile=mcp-app"

    legacy_product_gallery = next(
        resource
        for resource in resources
        if str(resource.uri) == server.LEGACY_PRODUCT_GALLERY_URI
    )
    assert legacy_product_gallery.mime_type == "text/html+skybridge"

    for compat_uri, compat_mime_type in (
        server.PRODUCT_COMPAT_GALLERY_RESOURCES
    ):
        compat_product_gallery = next(
            resource
            for resource in resources
            if str(resource.uri) == compat_uri
        )
        assert compat_product_gallery.mime_type == compat_mime_type

    resource_by_uri = {
        str(resource.uri): resource for resource in resources
    }
    for compat_uri, compat_mime_type in (
        server.COMPAT_GALLERY_RESOURCES
    ):
        assert resource_by_uri[compat_uri].mime_type == compat_mime_type
        compat_contents = await server.mcp.read_resource(compat_uri)
        assert "function showImages" in list(compat_contents)[0].content

    contents = await server.mcp.read_resource(
        server.KNOWLEDGE_GALLERY_URI
    )
    html = list(contents)[0].content
    assert "ui/notifications/tool-result" in html
    assert "data:${block.mimeType};base64" in html


@pytest.mark.asyncio
async def test_search_knowledge_returns_data_without_widget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = SearchResponse.model_validate(
        {
            "success": True,
            "status": "knowledge_found",
            "content": "Nội dung về logo Đông Hải.",
            "sources": [
                {
                    "document_id": "123",
                    "source_key": "employee-handbook",
                    "title": "Sổ tay nhân viên 2025",
                    "category": "handbook",
                    "chunk_index": 1,
                    "assets": [
                        {
                            "id": 2,
                            "asset_type": "image",
                            "page_number": 6,
                            "width": 334,
                            "height": 259,
                            "bbox": [0, 0, 334, 259],
                            "media_type": "image/png",
                            "url": "/api/v1/knowledge/assets/2",
                        },
                        {
                            "id": 2,
                            "asset_type": "image",
                            "page_number": 6,
                            "width": 334,
                            "height": 259,
                            "bbox": [0, 0, 334, 259],
                            "media_type": "image/png",
                            "url": "/api/v1/knowledge/assets/2",
                        },
                    ],
                }
            ],
            "elapsed_ms": 10.0,
        }
    )

    search_mock = AsyncMock(return_value=response)
    monkeypatch.setattr(server, "search_rag", search_mock)

    result = await server.search_knowledge(
        user_question_verbatim="Logo thương hiệu",
    )

    assert result.is_error is False
    assert result.structured_content is not None
    assert result.structured_content["status"] == "knowledge_found"
    assert result.structured_content["image_asset_ids"] == [2]
    assert not result.meta
    assert [item.type for item in result.content] == ["text"]
    request = search_mock.await_args.args[0]
    assert request.query == "Logo thương hiệu"
    assert request.top_k == 10
    assert request.categories is None
    assert request.doc_type_id is None
    assert request.group_ids is None


@pytest.mark.asyncio
async def test_show_knowledge_images_returns_one_gallery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset_mock = AsyncMock(
        side_effect=[
            (b"\x89PNG\r\n\x1a\nimage-2", "image/png"),
            (b"\x89PNG\r\n\x1a\nimage-3", "image/png"),
        ]
    )
    monkeypatch.setattr(server, "fetch_knowledge_asset", asset_mock)

    result = await server.show_knowledge_images(asset_ids=[2, 3, 2])

    assert result.meta is not None
    assert result.meta["openai/outputTemplate"] == (
        server.LEGACY_GALLERY_URI
    )
    assert [item.type for item in result.content] == ["image", "image"]
    assert len(result.structured_content["images"]) == 2
    assert asset_mock.await_count == 2


@pytest.mark.asyncio
async def test_get_knowledge_asset_returns_widget_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset_mock = AsyncMock(
        return_value=(b"\x89PNG\r\n\x1a\nimage-data", "image/png")
    )
    monkeypatch.setattr(server, "fetch_knowledge_asset", asset_mock)

    result = await server.get_knowledge_asset(asset_id=2)

    assert result.meta is not None
    assert result.meta["openai/outputTemplate"] == (
        server.LEGACY_GALLERY_URI
    )
    assert result.content[0].type == "image"
    assert result.structured_content["images"][0]["type"] == "image"


@pytest.mark.asyncio
async def test_search_products_returns_catalog_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = ProductSearchResponse.model_validate(
        {
            "products": [
                {
                    "point_id": "product-point",
                    "product_code": "S5921",
                    "title": "Sandal Kẹp Nữ Đông Hải",
                    "image_url": "https://cdn.example/product.jpg",
                }
            ],
            "limit": 3,
            "query": "S5921",
        }
    )
    search_mock = AsyncMock(return_value=response)
    monkeypatch.setattr(server, "search_products_api", search_mock)

    result = await server.search_products("S5921", 3)

    search_mock.assert_awaited_once_with("S5921", limit=3)
    assert result.structured_content["products"][0][
        "product_code"
    ] == "S5921"


@pytest.mark.asyncio
async def test_get_product_detail_filters_requested_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = ProductDetailResponse.model_validate(
        {
            "point_id": "product-point",
            "product": {
                "product_code": "S5921",
                "title": "Sandal Kẹp Nữ Đông Hải",
                "inventory_synced_at": "2026-10-19T02:09:49+00:00",
                "inventory_sync_complete": True,
                "public_info": {
                    "product_code": "S5921",
                    "product_name": "Sandal Kẹp Nữ Đông Hải",
                    "prices": [620000],
                    "colors": ["Cafe", "Đen"],
                    "available_sizes": ["37", "38"],
                    "image_urls": ["https://cdn.example/default.jpg"],
                    "image_urls_by_color": {
                        "Cafe": ["https://cdn.example/cafe.jpg"]
                    },
                },
                "variants": [
                    {
                        "color": "Cafe",
                        "size": "37",
                        "price": 620000,
                        "inventory_quantity": 11,
                        "available": True,
                    },
                    {
                        "color": "Đen",
                        "size": "38",
                        "price": 620000,
                        "inventory_quantity": 8,
                        "available": True,
                    },
                ],
                "detail": {
                    "product": {
                        "online_store_url": (
                            "https://shopdonghai.com/products/S5921"
                        )
                    }
                },
            },
        }
    )
    detail_mock = AsyncMock(return_value=response)
    image_mock = AsyncMock(
        return_value=(b"\xff\xd8\xffproduct-image", "image/jpeg")
    )
    monkeypatch.setattr(server, "fetch_product_detail", detail_mock)
    monkeypatch.setattr(server, "fetch_product_image", image_mock)

    result = await server.get_product_detail("S5921", "Cà phê", "37")
    data = result.structured_content

    detail_mock.assert_awaited_once_with("S5921")
    assert data["selection_found"] is True
    assert len(data["variants"]) == 2
    assert data["inventory_total"] == 19
    assert len(data["selected_variants"]) == 1
    assert data["selected_variants"][0]["inventory_quantity"] == 11
    assert data["selected_inventory_total"] == 11
    assert "image_urls" not in data
    assert "images" not in data
    assert data["displayed_image_count"] == 1
    assert len(result.meta["productImages"]) == 1
    assert result.meta["product"]["image_urls"] == [
        "https://cdn.example/cafe.jpg"
    ]
    assert [item.type for item in result.content] == ["text"]
    content_text = result.content[0].text
    assert "### Tổng quan sản phẩm" in content_text
    assert "Widget đã hiển thị 1 ảnh sản phẩm màu Cà phê" in content_text
    assert "### Tồn kho theo màu và size" in content_text
    assert "| Cafe | 37 | 11 | Còn hàng |" in content_text
    assert "| Đen | 38 | 8 | Còn hàng |" in content_text
    assert "**Tổng tồn kho toàn sản phẩm:** 19" in content_text
    assert "cdn.example" not in content_text
    assert result.meta["openai/outputTemplate"] == (
        server.LEGACY_PRODUCT_GALLERY_URI
    )


@pytest.mark.asyncio
async def test_product_images_are_merged_into_detail_tool() -> None:
    tools = await server.mcp.list_tools()
    tools_by_name = {tool.name: tool for tool in tools}

    assert "get_product_detail" in tools_by_name
    assert "show_product_images" not in tools_by_name
    assert tools_by_name["get_product_detail"].meta["ui"][
        "resourceUri"
    ] == server.PRODUCT_GALLERY_URI
