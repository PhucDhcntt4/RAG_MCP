from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

Category = Annotated[
    str,
    Field(min_length=1, max_length=100),
]


class SearchRequest(BaseModel):
    """Dữ liệu gửi đến RAG Search API."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=4000,
        description="Câu hỏi hoặc nội dung cần tìm kiếm",
    )

    categories: list[Category] | None = Field(
        default=None,
        max_length=20,
        description="Danh sách category cần giới hạn",
    )

    doc_type_id: int | None = Field(
        default=None,
        ge=1,
        description="ID loại tài liệu",
    )

    group_ids: list[int] | None = Field(
        default=None,
        max_length=20,
        description="Danh sách ID nhóm tài liệu",
    )

    top_k: int | None = Field(
        default=10,
        ge=1,
        le=20,
        description="Số đoạn tài liệu tối đa cần lấy",
    )


class SourceAsset(BaseModel):
    """Hình ảnh hoặc tài nguyên gắn với một nguồn."""

    model_config = ConfigDict(extra="ignore")

    id: int
    asset_type: str
    page_number: int
    width: int
    height: int
    bbox: list[float]
    media_type: str
    url: str
    caption: str | None = None


class SearchSource(BaseModel):
    model_config = ConfigDict(extra="ignore")

    document_id: str | None = None
    source_key: str
    title: str
    category: str
    heading: str | None = None
    section_path: list[str] = Field(default_factory=list)
    chunk_index: int

    similarity: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    bm25_rank: int | None = None
    vector_rank: int | None = None

    assets: list[SourceAsset] = Field(default_factory=list)

    @field_validator("document_id", mode="before")
    @classmethod
    def convert_document_id_to_string(
        cls,
        value: object,
    ) -> str | None:
        if value is None:
            return None

        return str(value)


class SearchResponse(BaseModel):
    """Kết quả trả về từ RAG Search API."""

    model_config = ConfigDict(extra="ignore")

    success: bool
    status: Literal[
        "knowledge_found",
        "knowledge_not_found",
    ]
    content: str
    sources: list[SearchSource]
    elapsed_ms: float
    image_asset_ids: list[int] = Field(
        default_factory=list,
        description=(
            "Danh sách ID hình ảnh duy nhất mà MCP nên hiển thị "
            "sau kết quả tìm kiếm."
        ),
    )

class ProductSearchItem(BaseModel):
    """Một sản phẩm trong kết quả tìm kiếm."""

    model_config = ConfigDict(extra="ignore")

    point_id: str
    product_code: str
    title: str
    product_type: str | None = None
    vendor: str | None = None
    status: str | None = None
    colors: str | None = None
    variant_count: int = 0
    image_count: int = 0
    embedding_count: int = 0
    ai_ready: bool = False
    image_url: str | None = None


class ProductSearchResponse(BaseModel):
    """Kết quả trả về từ GET /api/v1/products."""

    model_config = ConfigDict(extra="ignore")

    products: list[ProductSearchItem] = Field(default_factory=list)
    next_cursor: str | None = None
    limit: int
    query: str = ""
    product_type: str = ""
    status: str = ""


class ProductVariant(BaseModel):
    """Tồn kho và giá của một biến thể sản phẩm."""

    model_config = ConfigDict(extra="ignore")

    external_id: str | None = None
    legacy_id: str | None = None
    sku: str | None = None
    barcode: str | None = None
    variant_title: str | None = None
    color: str | None = None
    color_normalized: str | None = None
    size: str | None = None
    price: int | float | None = None
    compare_at_price: int | float | None = None
    inventory_quantity: int = 0
    available: bool = False


class ProductImage(BaseModel):
    """Thông tin một hình ảnh sản phẩm."""

    model_config = ConfigDict(extra="ignore")

    external_id: str | None = None
    color: str | None = None
    color_normalized: str | None = None
    source_url: str
    alt_text: str | None = None
    mime_type: str | None = None
    width: int | None = None
    height: int | None = None
    image_order: int | None = None
    is_featured: bool = False
    is_active: bool = True


class ProductPublicInfo(BaseModel):
    """Thông tin sản phẩm được phép trả cho người dùng."""

    model_config = ConfigDict(extra="ignore")

    product_code: str
    product_name: str
    product_type: str | None = None
    description: str | None = None
    material: str | None = None
    sole: str | None = None
    height: str | int | float | None = None
    status: str | None = None
    inventory_synced_at: str | None = None
    inventory_sync_complete: bool = False

    prices: list[int | float] = Field(default_factory=list)
    colors: list[str] = Field(default_factory=list)
    available_sizes: list[str] = Field(default_factory=list)
    image_urls: list[str] = Field(default_factory=list)
    image_urls_by_color: dict[str, list[str]] = Field(
        default_factory=dict
    )


class ProductStoreInfo(BaseModel):
    """Thông tin liên kết cửa hàng online."""

    model_config = ConfigDict(extra="ignore")

    online_store_url: str | None = None


class ProductDetailSection(BaseModel):
    """Khối detail lồng bên trong dữ liệu sản phẩm."""

    model_config = ConfigDict(extra="ignore")

    product: ProductStoreInfo = Field(
        default_factory=ProductStoreInfo
    )


class ProductData(BaseModel):
    """Thông tin đầy đủ của một sản phẩm."""

    model_config = ConfigDict(extra="ignore")

    product_code: str
    title: str
    product_type: str | None = None
    description: str | None = None
    vendor: str | None = None
    material: str | None = None
    sole: str | None = None
    height: str | int | float | None = None
    status: str | None = None

    inventory_source: str | None = None
    inventory_synced_at: str | None = None
    inventory_sync_complete: bool = False

    public_info: ProductPublicInfo
    variants: list[ProductVariant] = Field(default_factory=list)
    images: list[ProductImage] = Field(default_factory=list)
    detail: ProductDetailSection = Field(
        default_factory=ProductDetailSection
    )


class ProductDetailResponse(BaseModel):
    """Response của GET /api/v1/products/{product_code}."""

    model_config = ConfigDict(extra="ignore")

    point_id: str
    product: ProductData


class ProductEmbeddedImage(BaseModel):
    """Ảnh nhúng trong structuredContent của MCP tool."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["image"] = "image"
    data: str
    mimeType: str


class ProductDetailToolResponse(BaseModel):
    """Kết quả đã rút gọn mà MCP trả cho ChatGPT."""

    model_config = ConfigDict(extra="forbid")

    success: bool = True
    product_code: str
    title: str
    product_type: str | None = None
    description: str | None = None
    material: str | None = None
    sole: str | None = None
    height: str | int | float | None = None
    status: str | None = None

    requested_color: str | None = None
    requested_size: str | None = None
    selection_found: bool = True

    prices: list[int | float] = Field(default_factory=list)
    selected_prices: list[int | float] = Field(default_factory=list)
    colors: list[str] = Field(default_factory=list)
    available_sizes: list[str] = Field(default_factory=list)
    variants: list[ProductVariant] = Field(default_factory=list)
    selected_variants: list[ProductVariant] = Field(default_factory=list)
    inventory_total: int = 0
    selected_inventory_total: int = 0

    image_urls: list[str] = Field(default_factory=list)
    images: list[ProductEmbeddedImage] = Field(default_factory=list)
    displayed_image_count: int = 0
    online_store_url: str | None = None

    inventory_synced_at: str | None = None
    inventory_sync_complete: bool = False
