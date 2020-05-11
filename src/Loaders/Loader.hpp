
class TileLoader {
private:
    using copy_t = int4;
};

class GlmemToShmemTileLoader: public MemoryLoader {
public:
    ShmemTile* load(GlmemTile* gl_tile_ptr);
};
