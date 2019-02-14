
struct render_entry_3d_object
{
    v3 P;
    
    u32 VertexCount;
    b32 Optimized;
    b32 PhongShading;
    void *VertexData;
    void *ColorData;
    void *NormalData;
    void *UVData;
    void *EdgeMemory;
    loaded_bitmap *Bitmap;
};

struct edge_info
{
    s32 YMax;
    r32 XMin;
    r32 ZMin;
    r32 OneOverZMin;
    r32 Gradient;
    r32 ZGradient;
    r32 OneOverZGradient;
    s32 YMin;
    r32 UMin;
    r32 VMin;
    r32 UGradient;
    r32 VGradient;
    b32 Left;
    v4 MinColor;
    v4 ColorGradient;
    v3 MinNormal;
    v3 NormalGradient;
    edge_info *Next;
};

struct thread_edge_info
{
    
    r32 LeftXMin;
    r32 RightXMin;
    
    r32 LeftZMin;
    r32 RightZMin;
    
    r32 LeftOneOverZMin;
    r32 RightOneOverZMin;
    
    r32 LeftUMin;
    r32 RightUMin;
    
    r32 LeftVMin;
    r32 RightVMin;
    
    v4 LeftMinColor;
    v4 RightMinColor;
    
    v3 LeftMinNormal;
    v3 RightMinNormal;
    
};

struct line_render_work
{
    game_render_commands *Commands;
    loaded_bitmap *OutputTarget;
    loaded_bitmap *Bitmap;
    edge_info CurrentEdgeInList;
    edge_info NextEdgeInList;
    s32 RowIndex;
    b32 PhongShading;
};

struct buffer_line_render_work
{
    game_render_commands *Commands;
    loaded_bitmap *OutputTarget;
    loaded_bitmap *Bitmap;
    s32 RowIndex;
    b32 PhongShading;
    
    u32 EdgeCount;
    u32 Pad;
    thread_edge_info Edges;
    
};

struct model_render_work
{
    game_render_commands *Commands;
    loaded_bitmap *OutputTarget;
    loaded_bitmap *Bitmap;
    edge_info *EdgeMemory;
    u32 EdgeCount;
    b32 PhongShading;
};
