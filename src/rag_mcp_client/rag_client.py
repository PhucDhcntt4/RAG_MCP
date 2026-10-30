from urllib.parse import quote

import httpx
from pydantic import ValidationError

from .config import Settings, get_settings
from .models import (
    ProductDetailResponse,
    ProductSearchResponse,
    SearchRequest,
    SearchResponse,
)


class RagClientError(RuntimeError):
    """Lỗi an toàn được trả về khi không thể truy vấn RAG Service."""


async def search_knowledge(
    request: SearchRequest,
    *,
    settings: Settings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> SearchResponse:
    """Gửi yêu cầu tìm kiếm tới RAG Service."""

    current_settings = settings or get_settings()

    headers = {
        "Authorization": (
            f"Bearer {current_settings.search_api_key}"
        ),
        "Accept": "application/json",
    }

    payload = request.model_dump(exclude_none=True)

    try:
        async with httpx.AsyncClient(
            timeout=30.0,
            transport=transport,
        ) as client:
            response = await client.post(
                current_settings.rag_search_url,
                json=payload,
                headers=headers,
            )

            response.raise_for_status()

    except httpx.TimeoutException as exc:
        raise RagClientError(
            "RAG Service phản hồi quá thời gian."
        ) from exc

    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code

        raise RagClientError(
            f"RAG Service trả về HTTP {status_code}."
        ) from exc

    except httpx.RequestError as exc:
        raise RagClientError(
            "Không thể kết nối đến RAG Service."
        ) from exc

    try:
        response_data = response.json()
        return SearchResponse.model_validate(response_data)

    except ValueError as exc:
        raise RagClientError(
            "RAG Service không trả về JSON hợp lệ."
        ) from exc

    except ValidationError as exc:
        raise RagClientError(
            "Response của RAG Service không đúng schema."
        ) from exc


async def get_knowledge_asset(
    asset_id: int,
    *,
    settings: Settings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[bytes, str]:
    """Tải một hình ảnh đính kèm từ RAG Service."""

    if asset_id < 1:
        raise RagClientError("Asset ID không hợp lệ.")

    current_settings = settings or get_settings()
    search_url = httpx.URL(current_settings.rag_search_url)
    asset_url = search_url.copy_with(
        path=f"/api/v1/knowledge/assets/{asset_id}",
        query=None,
        fragment=None,
    )

    headers = {
        "Authorization": (
            f"Bearer {current_settings.search_api_key}"
        ),
        "Accept": "image/*",
    }

    try:
        async with httpx.AsyncClient(
            timeout=30.0,
            transport=transport,
        ) as client:
            response = await client.get(
                asset_url,
                headers=headers,
            )

            response.raise_for_status()

    except httpx.TimeoutException as exc:
        raise RagClientError(
            "RAG Service phản hồi ảnh quá thời gian."
        ) from exc

    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code

        raise RagClientError(
            f"RAG Service trả về HTTP {status_code} khi tải ảnh."
        ) from exc

    except httpx.RequestError as exc:
        raise RagClientError(
            "Không thể tải ảnh từ RAG Service."
        ) from exc

    media_type = response.headers.get(
        "Content-Type",
        "",
    ).split(";", 1)[0].lower()

    if not media_type.startswith("image/"):
        raise RagClientError(
            "Asset nhận được không phải là hình ảnh."
        )

    image_data = response.content

    if len(image_data) > 10 * 1024 * 1024:
        raise RagClientError(
            "Hình ảnh vượt quá giới hạn 10 MB."
        )

    return image_data, media_type

async def search_products(
    query: str,
    *,
    limit: int = 6,
    settings: Settings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ProductSearchResponse:
    """Tìm sản phẩm từ Product API của RAG Service."""

    current_settings = settings or get_settings()

    headers = {
        "Authorization": (
            f"Bearer {current_settings.search_api_key}"
        ),
        "Accept": "application/json",
    }

    params = {
        "q": query,
        "limit": limit,
    }

    try:
        async with httpx.AsyncClient(
            timeout=30.0,
            transport=transport,
        ) as client:
            response = await client.get(
                current_settings.rag_products_url,
                params=params,
                headers=headers,
            )

            response.raise_for_status()

    except httpx.TimeoutException as exc:
        raise RagClientError(
            "Product API phản hồi quá thời gian."
        ) from exc

    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code

        raise RagClientError(
            f"Product API trả về HTTP {status_code}."
        ) from exc

    except httpx.RequestError as exc:
        raise RagClientError(
            "Không thể kết nối đến Product API."
        ) from exc

    try:
        return ProductSearchResponse.model_validate(
            response.json()
        )

    except ValueError as exc:
        raise RagClientError(
            "Product API không trả về JSON hợp lệ."
        ) from exc

    except ValidationError as exc:
        raise RagClientError(
            "Kết quả tìm sản phẩm không đúng schema."
        ) from exc

async def get_latest_products(
    *,
    limit: int = 10,
    page: int = 1,
    date_from: str | None = None,
    date_to: str | None = None,
    product_type: str | None = None,
    settings: Settings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict:
    """Lấy danh sách sản phẩm ACTIVE mới theo new_at từ RAG Service."""
    current_settings = settings or get_settings()
    params = {
        "limit": limit, "page": page, "date_from": date_from,
        "date_to": date_to, "product_type": product_type,
    }
    params = {key: value for key, value in params.items() if value is not None}
    try:
        async with httpx.AsyncClient(timeout=30.0, transport=transport) as client:
            response = await client.get(
                current_settings.rag_products_url.rstrip("/") + "/latest",
                params=params,
                headers={
                    "Authorization": f"Bearer {current_settings.search_api_key}",
                    "Accept": "application/json",
                },
            )
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise RagClientError("Product API phản hồi quá thời gian.") from exc
    except httpx.HTTPStatusError as exc:
        raise RagClientError(
            f"Product API trả về HTTP {exc.response.status_code}."
        ) from exc
    except httpx.RequestError as exc:
        raise RagClientError("Không thể kết nối đến Product API.") from exc
    try:
        result = response.json()
    except ValueError as exc:
        raise RagClientError("Product API không trả về JSON hợp lệ.") from exc
    if not isinstance(result, dict) or not isinstance(result.get("products"), list):
        raise RagClientError("Dữ liệu sản phẩm mới không đúng định dạng.")
    return result


async def get_product_detail(
    product_code: str,
    *,
    settings: Settings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ProductDetailResponse:
    """Lấy thông tin, giá, tồn kho và ảnh của sản phẩm."""

    current_settings = settings or get_settings()

    normalized_code = product_code.strip()

    if not normalized_code:
        raise RagClientError("Mã sản phẩm không hợp lệ.")

    encoded_code = quote(normalized_code, safe="")
    detail_url = (
        f"{current_settings.rag_products_url.rstrip('/')}"
        f"/{encoded_code}"
    )

    headers = {
        "Authorization": (
            f"Bearer {current_settings.search_api_key}"
        ),
        "Accept": "application/json",
    }

    try:
        async with httpx.AsyncClient(
            timeout=30.0,
            transport=transport,
        ) as client:
            response = await client.get(
                detail_url,
                headers=headers,
            )

            response.raise_for_status()

    except httpx.TimeoutException as exc:
        raise RagClientError(
            "Product API phản hồi quá thời gian."
        ) from exc

    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code

        if status_code == 404:
            raise RagClientError(
                f"Không tìm thấy sản phẩm {normalized_code}."
            ) from exc

        raise RagClientError(
            f"Product API trả về HTTP {status_code}."
        ) from exc

    except httpx.RequestError as exc:
        raise RagClientError(
            "Không thể kết nối đến Product API."
        ) from exc

    try:
        return ProductDetailResponse.model_validate(
            response.json()
        )

    except ValueError as exc:
        raise RagClientError(
            "Product API không trả về JSON hợp lệ."
        ) from exc

    except ValidationError as exc:
        raise RagClientError(
            "Chi tiết sản phẩm không đúng schema."
        ) from exc


async def get_product_image(
    image_url: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[bytes, str]:
    """Tải ảnh sản phẩm từ CDN Shopify đã được cho phép."""

    url = httpx.URL(image_url)
    allowed_hosts = {"cdn.shopify.com"}

    if url.scheme != "https" or url.host not in allowed_hosts:
        raise RagClientError("URL ảnh sản phẩm không được phép.")

    # Widget chỉ cần ảnh xem trước. Shopify tự tối ưu ảnh theo chiều
    # rộng này, giúp giảm đáng kể dung lượng MCP response trong khi
    # vẫn giữ chất lượng tốt trong widget ChatGPT.
    url = url.copy_set_param("width", "600")

    try:
        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            transport=transport,
            trust_env=False,
        ) as client:
            response = await client.get(
                url,
                headers={"Accept": "image/*"},
            )
            response.raise_for_status()

    except httpx.TimeoutException as exc:
        raise RagClientError(
            "CDN phản hồi ảnh sản phẩm quá thời gian."
        ) from exc

    except httpx.HTTPStatusError as exc:
        raise RagClientError(
            f"CDN ảnh sản phẩm trả về HTTP {exc.response.status_code}."
        ) from exc

    except httpx.RequestError as exc:
        raise RagClientError(
            "Không thể tải ảnh sản phẩm từ CDN."
        ) from exc

    final_url = response.url
    if final_url.scheme != "https" or final_url.host not in allowed_hosts:
        raise RagClientError("CDN chuyển hướng đến URL không được phép.")

    media_type = response.headers.get(
        "Content-Type",
        "",
    ).split(";", 1)[0].lower()

    if not media_type.startswith("image/"):
        raise RagClientError(
            "Dữ liệu nhận được không phải là ảnh sản phẩm."
        )

    image_data = response.content
    if len(image_data) > 10 * 1024 * 1024:
        raise RagClientError(
            "Ảnh sản phẩm vượt quá giới hạn 10 MB."
        )

    return image_data, media_type
