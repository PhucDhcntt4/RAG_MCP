
import asyncio
import json
from importlib.resources import files
from pathlib import Path
from typing import Annotated
import unicodedata

from mcp.server.apps import Apps, ResourceCsp
from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.resources import TextResource
from mcp.types import (
    CallToolResult,
    ContentBlock,
    TextContent,
    ToolAnnotations,
)
from pydantic import Field

from .config import get_settings
from .models import (
    ProductEmbeddedImage,
    ProductDetailToolResponse,
    ProductSearchResponse,
    SearchRequest,
    SearchResponse,
)
from .rag_client import (
    RagClientError,
    get_knowledge_asset as fetch_knowledge_asset,
    get_product_detail as fetch_product_detail,
    get_latest_products as fetch_latest_products,
    get_product_image as fetch_product_image,
    search_knowledge as search_rag,
    search_products as search_products_api,
)

settings = get_settings()
RAG_TOP_K = 10

COLOR_ALIASES = {
    "caphe": "cafe",
    "coffee": "cafe",
}


def normalize_color_name(value: str) -> str:
    """Chuẩn hóa tên màu để khớp cách viết có dấu/không dấu."""

    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_accents = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    ).replace("đ", "d")
    compact = "".join(
        character
        for character in without_accents
        if character.isalnum()
    )
    return COLOR_ALIASES.get(compact, compact)


def format_vnd(value: int | float) -> str:
    """Format a product price for a Vietnamese customer."""

    rounded_value = round(value)
    return f"{rounded_value:,}".replace(",", ".") + " đ"


def format_product_detail_for_model(
    result: ProductDetailToolResponse,
) -> str:
    """Build a complete, ready-to-use product answer for the model."""

    price_text = (
        " – ".join(format_vnd(price) for price in result.prices)
        if result.prices
        else "Chưa có dữ liệu"
    )
    colors_text = ", ".join(result.colors) or "Chưa có dữ liệu"
    sizes_text = (
        ", ".join(result.available_sizes)
        or "Chưa có dữ liệu"
    )
    lines = [
        f"### {result.product_code} — {result.title}",
        "",
        "### Tổng quan sản phẩm",
        f"- **Giá:** {price_text}",
        f"- **Loại sản phẩm:** {result.product_type or 'Chưa có dữ liệu'}",
        f"- **Mô tả:** {result.description or 'Chưa có dữ liệu'}",
        f"- **Chất liệu:** {result.material or 'Chưa có dữ liệu'}",
        f"- **Đế:** {result.sole or 'Chưa có dữ liệu'}",
        f"- **Chiều cao:** {result.height or 'Chưa có dữ liệu'}",
        f"- **Màu sắc:** {colors_text}",
        f"- **Kích thước:** {sizes_text}",
        f"- **Trạng thái:** {result.status or 'Chưa có dữ liệu'}",
    ]
    if result.displayed_image_count > 0:
        image_scope = (
            f" màu {result.requested_color}"
            if result.requested_color
            else ""
        )
        lines.append(
            "- **Hình ảnh:** Widget đã hiển thị "
            f"{result.displayed_image_count} ảnh sản phẩm{image_scope}."
        )
    else:
        lines.append(
            "- **Hình ảnh:** Widget không nhận được ảnh sản phẩm."
        )

    if result.requested_color or result.requested_size:
        selection = ", ".join(
            value
            for value in (
                (
                    f"màu {result.requested_color}"
                    if result.requested_color
                    else None
                ),
                (
                    f"size {result.requested_size}"
                    if result.requested_size
                    else None
                ),
            )
            if value
        )
        lines.extend(
            [
                "",
                f"### Lựa chọn đang xem: {selection}",
            ]
        )
        if not result.selection_found:
            lines.append("Không tìm thấy biến thể phù hợp.")
        else:
            lines.extend(
                [
                    "| Màu | Size | Số lượng | Trạng thái |",
                    "|---|---:|---:|---|",
                ]
            )
            for variant in result.selected_variants:
                availability = (
                    "Còn hàng"
                    if (
                        variant.available
                        and variant.inventory_quantity > 0
                    )
                    else "Hết hàng"
                )
                lines.append(
                    "| "
                    f"{variant.color or '-'} | {variant.size or '-'} | "
                    f"{variant.inventory_quantity} | {availability} |"
                )
            lines.append(
                "\n**Tổng tồn kho lựa chọn:** "
                f"{result.selected_inventory_total} sản phẩm"
            )

    lines.extend(
        [
            "",
            "### Tồn kho theo màu và size",
            "| Màu | Size | Số lượng | Trạng thái |",
            "|---|---:|---:|---|",
        ]
    )
    for variant in result.variants:
        availability = (
            "Còn hàng"
            if variant.available and variant.inventory_quantity > 0
            else "Hết hàng"
        )
        lines.append(
            "| "
            f"{variant.color or '-'} | {variant.size or '-'} | "
            f"{variant.inventory_quantity} | {availability} |"
        )

    lines.append(
        f"\n**Tổng tồn kho toàn sản phẩm:** "
        f"{result.inventory_total} sản phẩm"
    )
    if result.inventory_synced_at:
        lines.append(
            f"**Cập nhật tồn kho:** {result.inventory_synced_at}"
        )
    if not result.inventory_sync_complete:
        lines.append(
            "**Lưu ý:** Dữ liệu tồn kho có thể chưa đồng bộ xong."
        )
    if result.online_store_url:
        lines.extend(
            [
                "",
                f"[Xem và mua sản phẩm]({result.online_store_url})",
            ]
        )

    return "\n".join(lines)

KNOWLEDGE_GALLERY_URI = "ui://rag-mcp-client/knowledge-gallery-v6.html"
LEGACY_GALLERY_URI = (
    "ui://rag-mcp-client/knowledge-gallery-legacy-v6.html"
)
PRODUCT_GALLERY_URI = "ui://rag-mcp-client/product-gallery-v5.html"
LEGACY_PRODUCT_GALLERY_URI = (
    "ui://rag-mcp-client/product-gallery-legacy-v5.html"
)
PRODUCT_COMPAT_GALLERY_RESOURCES = (
    (
        "ui://rag-mcp-client/product-gallery-v4.html",
        "text/html;profile=mcp-app",
    ),
    (
        "ui://rag-mcp-client/product-gallery-legacy-v4.html",
        "text/html+skybridge",
    ),
    (
        "ui://rag-mcp-client/product-gallery-v3.html",
        "text/html;profile=mcp-app",
    ),
    (
        "ui://rag-mcp-client/product-gallery-legacy-v3.html",
        "text/html+skybridge",
    ),
    (
        "ui://rag-mcp-client/product-gallery-v2.html",
        "text/html;profile=mcp-app",
    ),
    (
        "ui://rag-mcp-client/product-gallery-legacy-v2.html",
        "text/html+skybridge",
    ),
    (
        "ui://rag-mcp-client/product-gallery-v1.html",
        "text/html;profile=mcp-app",
    ),
    (
        "ui://rag-mcp-client/product-gallery-legacy-v1.html",
        "text/html+skybridge",
    ),
)
COMPAT_GALLERY_RESOURCES = (
    (
        "ui://rag-mcp-client/knowledge-gallery-v5.html",
        "text/html;profile=mcp-app",
    ),
    (
        "ui://rag-mcp-client/knowledge-gallery-legacy-v5.html",
        "text/html+skybridge",
    ),
    (
        "ui://rag-mcp-client/knowledge-gallery-v4.html",
        "text/html;profile=mcp-app",
    ),
    (
        "ui://rag-mcp-client/knowledge-gallery-legacy-v4.html",
        "text/html+skybridge",
    ),
)
KNOWLEDGE_GALLERY_HTML = (
    files("rag_mcp_client")
    .joinpath("ui/knowledge_gallery.html")
    .read_text(encoding="utf-8")
)
PRODUCT_GALLERY_HTML = (
    files("rag_mcp_client")
    .joinpath("ui/product_gallery.html")
    .read_text(encoding="utf-8")
)
LOCAL_SERVER_INSTRUCTIONS_PATH = Path(
    "src/rag_mcp_client/prompt/server_instructions.txt"
)


def load_server_instructions() -> str:
    """Load editable local instructions, then fall back to package data."""

    if LOCAL_SERVER_INSTRUCTIONS_PATH.is_file():
        return LOCAL_SERVER_INSTRUCTIONS_PATH.read_text(
            encoding="utf-8"
        ).strip()

    return (
        files("rag_mcp_client")
        .joinpath("prompt/server_instructions.txt")
        .read_text(encoding="utf-8")
        .strip()
    )


SERVER_INSTRUCTIONS = load_server_instructions()

OPENAI_WIDGET_META = {
    "openai/outputTemplate": LEGACY_GALLERY_URI,
    "openai/toolInvocation/invoking": "Đang tìm trong kho tri thức…",
    "openai/toolInvocation/invoked": "Đã tìm xong.",
    "openai/widgetAccessible": True,
}
PRODUCT_WIDGET_META = {
    "openai/outputTemplate": LEGACY_PRODUCT_GALLERY_URI,
    "openai/toolInvocation/invoking": "Đang tải ảnh sản phẩm…",
    "openai/toolInvocation/invoked": "Đã tải ảnh sản phẩm.",
    "openai/widgetAccessible": True,
}

apps = Apps()
apps.add_html_resource(
    KNOWLEDGE_GALLERY_URI,
    KNOWLEDGE_GALLERY_HTML,
    name="Dong Hai knowledge image gallery",
    title="Hình ảnh từ kho tri thức Đông Hải",
    description="Hiển thị các hình ảnh được tìm thấy trong kho tri thức.",
    prefers_border=True,
)
apps.add_html_resource(
    PRODUCT_GALLERY_URI,
    PRODUCT_GALLERY_HTML,
    name="Dong Hai product gallery",
    title="Hình ảnh sản phẩm Đông Hải",
    description="Hiển thị hình ảnh của sản phẩm Đông Hải.",
    csp=ResourceCsp(
        resourceDomains=["https://cdn.shopify.com"],
    ),
    prefers_border=True,
)

async def search_knowledge(
    user_question_verbatim: Annotated[
        str,
        Field(
            min_length=1,
            max_length=4000,
            description=(
                "BẮT BUỘC sao chép nguyên văn tin nhắn hiện tại của "
                "người dùng. Không diễn giải, không thêm từ khóa, "
                "không mở rộng ý định và không ghép thêm nội dung. "
                "Ví dụ: người dùng hỏi 'thời gian làm việc' thì giá "
                "trị phải đúng là 'thời gian làm việc'."
            ),
        ),
    ],
) -> Annotated[CallToolResult, SearchResponse]:
    """
Tìm kiếm cơ sở tri thức riêng được kết nối.

Sử dụng công cụ này cho các câu hỏi về chính sách, bảo hành, thông tin nhân viên, chăm sóc khách hàng, hoặc các kiến thức từ tài liệu khác. Không sử dụng nó cho các yêu cầu về danh mục sản phẩm, giá cả, kích thước, màu sắc, tồn kho, hoặc hình ảnh sản phẩm; thay vào đó hãy dùng search_products.

Sao chép chính xác tin nhắn hiện tại của người dùng vào user_question_verbatim. Không bao giờ diễn đạt lại, mở rộng, tối ưu hoá, hoặc thêm các ý định liên quan. Máy chủ luôn tìm kiếm với top_k=10.

Sau mỗi lần gọi, kiểm tra image_asset_ids. Nếu không rỗng, gọi show_knowledge_images một lần với những ID đó trước khi trả lời.
    """

    request = SearchRequest(
        query=user_question_verbatim,
        top_k=RAG_TOP_K,
    )

    try:
        result = await search_rag(request)

    except RagClientError as exc:
        raise ToolError(str(exc)) from exc

    image_asset_ids: list[int] = []
    seen_asset_ids: set[int] = set()
    for source in result.sources:
        for asset in source.assets:
            if asset.asset_type != "image" or asset.id in seen_asset_ids:
                continue
            image_asset_ids.append(asset.id)
            seen_asset_ids.add(asset.id)
            if len(image_asset_ids) >= 3:
                break
        if len(image_asset_ids) >= 3:
            break

    result = result.model_copy(
        update={"image_asset_ids": image_asset_ids}
    )

    content: list[ContentBlock] = [
        TextContent(
            type="text",
            text=result.model_dump_json(indent=2),
        )
    ]

    return CallToolResult(
        content=content,
        structuredContent=result.model_dump(mode="json"),
    )


async def search_products(
    product_query: Annotated[
        str,
        Field(
            min_length=1,
            max_length=200,
            description=(
                "Mã sản phẩm hoặc cụm từ mô tả sản phẩm cần tìm. "
                "Ví dụ: 'S5921', 'sandal nữ', 'giày cao gót'. "
                "Không thêm các câu dẫn như 'hãy tìm giúp tôi'."
            ),
        ),
    ],
    limit: Annotated[
        int,
        Field(
            ge=1,
            le=10,
            description="Số sản phẩm tối đa cần trả về.",
        ),
    ] = 6,
) -> Annotated[CallToolResult, ProductSearchResponse]:
    """
Tìm trong danh mục sản phẩm Đông Hải.
Sử dụng công cụ này khi người dùng muốn tìm, duyệt, hoặc xác định sản phẩm. Truy vấn chỉ nên chứa mã sản phẩm, tên sản phẩm, hoặc loại sản phẩm chính. Sử dụng get_product_detail sau đó khi cần giá, kích thước, màu sắc, tồn kho, hình ảnh, hoặc URL mua hàng.
    """

    try:
        result = await search_products_api(
            product_query,
            limit=limit,
        )

    except RagClientError as exc:
        raise ToolError(str(exc)) from exc

    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=result.model_dump_json(indent=2),
            )
        ],
        structuredContent=result.model_dump(mode="json"),
    )


async def get_latest_products(
    limit: Annotated[int, Field(ge=1, le=50, description="Số sản phẩm mỗi trang (1-50).")] = 10,
    page: Annotated[int, Field(ge=1, description="Trang cần lấy, bắt đầu từ 1.")] = 1,
    date_from: Annotated[str | None, Field(
        max_length=40,
        description="Ngày bắt đầu có múi giờ ISO 8601, ví dụ 2026-10-01T00:00:00+07:00.",
    )] = None,
    date_to: Annotated[str | None, Field(
        max_length=40,
        description="Ngày kết thúc KHÔNG bao gồm, ví dụ 2026-11-01T00:00:00+07:00.",
    )] = None,
    product_type: Annotated[str | None, Field(
        max_length=200, description="Lọc theo loại sản phẩm, nếu cần."
    )] = None,
) -> CallToolResult:
    """Liệt kê sản phẩm Đông Hải mới nhất hoặc mới theo tháng/khoảng ngày.

    Sử dụng ngày new_at (ngày đăng sản phẩm, không nhất thiết là ngày ra mắt).
    Dùng phân trang để lấy đủ sản phẩm nếu total_pages > page.
    """
    try:
        result = await fetch_latest_products(
            limit=limit, page=page, date_from=date_from,
            date_to=date_to, product_type=product_type,
        )
    except RagClientError as exc:
        raise ToolError(str(exc)) from exc
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(result, ensure_ascii=False))],
        structuredContent=result,
    )


async def get_product_detail(
    product_code: Annotated[
        str,
        Field(
            min_length=1,
            max_length=100,
            description=(
                "Mã sản phẩm chính xác nhận được từ search_products. "
                "Ví dụ: S5921."
            ),
        ),
    ],
    color: Annotated[
        str | None,
        Field(
            max_length=100,
            description=(
                "Màu người dùng muốn kiểm tra, ví dụ: Kem hoặc Đen."
            ),
        ),
    ] = None,
    size: Annotated[
        str | None,
        Field(
            max_length=50,
            description=(
                "Kích thước người dùng muốn kiểm tra, ví dụ: 37."
            ),
        ),
    ] = None,
) -> Annotated[CallToolResult, ProductDetailToolResponse]:
    """
Lấy giá, mô tả, màu sắc, kích thước, tồn kho chi tiết theo từng biến thể, hình ảnh và URL mua hàng cho một sản phẩm Dong Hai cụ thể.

Mã sản phẩm phải do người dùng cung cấp hoặc từ kết quả search_products trước đó. Truyền màu sắc và kích thước khi người dùng hỏi về một biến thể cụ thể. Mảng variants luôn chứa toàn bộ biến thể của sản phẩm để trình bày tổng quan; selected_variants chỉ chứa các biến thể khớp với màu/size được yêu cầu. Mỗi phần tử trong hai mảng là một dòng tồn kho gồm màu, size, inventory_quantity và available.
    """

    try:
        raw_result = await fetch_product_detail(product_code)

    except RagClientError as exc:
        raise ToolError(str(exc)) from exc

    product = raw_result.product
    requested_color = color.strip() if color else None
    requested_size = size.strip() if size else None
    matching_variants = list(product.variants)

    if requested_color:
        normalized_color = normalize_color_name(requested_color)
        matching_variants = [
            variant
            for variant in matching_variants
            if (
                variant.color is not None
                and normalize_color_name(variant.color)
                == normalized_color
            )
        ]

    if requested_size:
        normalized_size = requested_size.casefold()
        matching_variants = [
            variant
            for variant in matching_variants
            if (
                variant.size is not None
                and variant.size.casefold() == normalized_size
            )
        ]

    selection_found = bool(matching_variants)
    image_urls = product.public_info.image_urls

    if requested_color:
        normalized_color = normalize_color_name(requested_color)

        for actual_color, color_images in (
            product.public_info.image_urls_by_color.items()
        ):
            if normalize_color_name(actual_color) == normalized_color:
                image_urls = color_images
                break

    image_urls = image_urls[:5]
    embedded_images: list[ProductEmbeddedImage] = []

    fetched_images = await asyncio.gather(
        *(
            fetch_product_image(image_url)
            for image_url in image_urls[:3]
        ),
        return_exceptions=True,
    )

    for fetched_image in fetched_images:
        if isinstance(fetched_image, RagClientError):
            continue
        if isinstance(fetched_image, BaseException):
            raise fetched_image

        image_data, media_type = fetched_image

        image = Image(
            data=image_data,
            format=media_type.removeprefix("image/"),
        ).to_image_content()
        image_payload = image.model_dump(
            by_alias=True,
            mode="json",
        )
        embedded_images.append(
            ProductEmbeddedImage(
                data=image_payload["data"],
                mimeType=image_payload["mimeType"],
            )
        )

    prices = sorted(
        {
            variant.price
            for variant in product.variants
            if variant.price is not None
        }
    )
    selected_prices = sorted(
        {
            variant.price
            for variant in matching_variants
            if variant.price is not None
        }
    )

    result = ProductDetailToolResponse(
        product_code=product.product_code,
        title=product.title,
        product_type=product.product_type,
        description=(
            product.public_info.description
            or product.description
        ),
        material=(
            product.public_info.material
            or product.material
        ),
        sole=product.public_info.sole or product.sole,
        height=(
            product.public_info.height
            or product.height
        ),
        status=product.status,
        requested_color=requested_color,
        requested_size=requested_size,
        selection_found=selection_found,
        prices=prices,
        selected_prices=selected_prices,
        colors=product.public_info.colors,
        available_sizes=product.public_info.available_sizes,
        variants=product.variants,
        selected_variants=matching_variants,
        inventory_total=sum(
            max(variant.inventory_quantity, 0)
            for variant in product.variants
        ),
        selected_inventory_total=sum(
            max(variant.inventory_quantity, 0)
            for variant in matching_variants
        ),
        image_urls=image_urls,
        images=embedded_images,
        displayed_image_count=len(embedded_images),
        online_store_url=(
            product.detail.product.online_store_url
        ),
        inventory_synced_at=product.inventory_synced_at,
        inventory_sync_complete=(
            product.inventory_sync_complete
        ),
    )

    return CallToolResult(
        _meta={
            **PRODUCT_WIDGET_META,
            "product": {
                "product_code": result.product_code,
                "title": result.title,
                "color": result.requested_color,
                "image_urls": result.image_urls[:3],
            },
            "productImages": [
                image.model_dump(mode="json")
                for image in embedded_images
            ],
        },
        content=[
            TextContent(
                type="text",
                text=format_product_detail_for_model(result),
            )
        ],
        structuredContent=result.model_dump(
            mode="json",
            exclude={"images", "image_urls"},
        ),
    )


mcp = MCPServer(
    name="rag-mcp-client",
    instructions=SERVER_INSTRUCTIONS,
    extensions=[apps],
)
mcp.tool(
    title="Search private knowledge base",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)(search_knowledge)
mcp.tool(
    title="Search Dong Hai products",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)(search_products)
mcp.tool(
    title="List latest Dong Hai products by date",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)(get_latest_products)
mcp.tool(
    meta={
        **PRODUCT_WIDGET_META,
        "ui": {"resourceUri": PRODUCT_GALLERY_URI},
    },
    title="Get Dong Hai product details",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)(get_product_detail)
mcp.add_resource(
    TextResource(
        uri=LEGACY_GALLERY_URI,
        name="Dong Hai knowledge image gallery (ChatGPT)",
        title="Hình ảnh từ kho tri thức Đông Hải",
        description=(
            "Phiên bản widget tương thích với ChatGPT."
        ),
        mime_type="text/html+skybridge",
        meta=OPENAI_WIDGET_META,
        text=KNOWLEDGE_GALLERY_HTML,
    )
)
mcp.add_resource(
    TextResource(
        uri=LEGACY_PRODUCT_GALLERY_URI,
        name="Dong Hai product gallery (ChatGPT)",
        title="Hình ảnh sản phẩm Đông Hải",
        description=(
            "Phiên bản widget sản phẩm tương thích với ChatGPT."
        ),
        mime_type="text/html+skybridge",
        meta={
            **PRODUCT_WIDGET_META,
            "openai/widgetCSP": {
                "resource_domains": ["https://cdn.shopify.com"],
            },
        },
        text=PRODUCT_GALLERY_HTML,
    )
)
for compat_uri, compat_mime_type in PRODUCT_COMPAT_GALLERY_RESOURCES:
    mcp.add_resource(
        TextResource(
            uri=compat_uri,
            name="Dong Hai product gallery (compatibility)",
            title="Hình ảnh sản phẩm Đông Hải",
            description=(
                "Alias tương thích cho widget sản phẩm đã được cache."
            ),
            mime_type=compat_mime_type,
            meta=(
                {
                    "ui": {
                        "prefersBorder": True,
                        "csp": {
                            "resourceDomains": [
                                "https://cdn.shopify.com"
                            ],
                        },
                    },
                }
                if compat_mime_type == "text/html;profile=mcp-app"
                else {
                    **PRODUCT_WIDGET_META,
                    "openai/widgetCSP": {
                        "resource_domains": [
                            "https://cdn.shopify.com"
                        ],
                    },
                }
            ),
            text=PRODUCT_GALLERY_HTML,
        )
    )
for compat_uri, compat_mime_type in COMPAT_GALLERY_RESOURCES:
    mcp.add_resource(
        TextResource(
            uri=compat_uri,
            name="Dong Hai knowledge image gallery (compatibility)",
            title="Hình ảnh từ kho tri thức Đông Hải",
            description="Alias tương thích cho widget đã được cache.",
            mime_type=compat_mime_type,
            meta=(
                {"ui": {"prefersBorder": True}}
                if compat_mime_type == "text/html;profile=mcp-app"
                else OPENAI_WIDGET_META
            ),
            text=KNOWLEDGE_GALLERY_HTML,
        )
    )


@mcp.tool(
    meta={
        **OPENAI_WIDGET_META,
        "ui": {"resourceUri": KNOWLEDGE_GALLERY_URI},
    },
    title="Show knowledge-base images",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def show_knowledge_images(
    asset_ids: Annotated[
        list[int],
        Field(
            min_length=1,
            max_length=3,
            description=(
                "Danh sách image_asset_ids do search_knowledge trả về."
            ),
        ),
    ],
) -> CallToolResult:
    """
Hiển thị tất cả các tài sản hình ảnh được trả về từ tìm kiếm kiến thức trước đó.

Gọi công cụ này đúng một lần mỗi khi search_knowledge trả về danh sách image_asset_ids không rỗng. Chuyển danh sách đó mà không sử dụng các URL tài sản nội bộ.
    """

    images: list[ContentBlock] = []
    seen_asset_ids: set[int] = set()

    for asset_id in asset_ids:
        if asset_id in seen_asset_ids:
            continue
        seen_asset_ids.add(asset_id)

        try:
            image_data, media_type = await fetch_knowledge_asset(asset_id)
        except RagClientError:
            continue

        image = Image(
            data=image_data,
            format=media_type.removeprefix("image/"),
        ).to_image_content()
        images.append(image)

    if not images:
        raise ToolError("Không thể tải các hình ảnh đã tìm thấy.")

    return CallToolResult(
        _meta=OPENAI_WIDGET_META,
        content=images,
        structuredContent={
            "images": [
                image.model_dump(by_alias=True, mode="json")
                for image in images
            ],
        },
    )


@mcp.tool(
    meta={
        **OPENAI_WIDGET_META,
        "ui": {"resourceUri": KNOWLEDGE_GALLERY_URI},
    },
    title="Get knowledge-base image",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def get_knowledge_asset(
    asset_id: Annotated[
        int,
        Field(
            ge=1,
            description=(
                "ID của hình ảnh trong trường assets "
                "do search_knowledge trả về."
            ),
        ),
    ],
) -> CallToolResult:
    """
Hiển thị Fetch và hiển thị một hình ảnh được trả về bởi search_knowledge.  

Chỉ sử dụng công cụ này với một ID tài sản từ kết quả tìm kiếm cơ sở tri thức trước đó, khi người dùng yêu cầu xem hình ảnh liên quan.
    """

    try:
        image_data, media_type = await fetch_knowledge_asset(asset_id)

    except RagClientError as exc:
        raise ToolError(str(exc)) from exc

    image_format = media_type.removeprefix("image/")
    image = Image(data=image_data, format=image_format).to_image_content()
    return CallToolResult(
        _meta=OPENAI_WIDGET_META,
        content=[image],
        structuredContent={
            "images": [image.model_dump(by_alias=True, mode="json")],
        },
    )


def main() -> None:
    mcp.run(
        transport="streamable-http",
        host=settings.mcp_host,
        port=settings.mcp_port,
        streamable_http_path=settings.mcp_path,
        stateless_http=True,
        json_response=True,
    )


if __name__ == "__main__":
    main()
