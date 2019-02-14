
internal void
MergeSort(u32 Count, edge_info *First, edge_info *Temp)
{
    if(Count == 1)
    {

    }
    else if(Count == 2)
    {
        edge_info *EntryA = First;
        edge_info *EntryB = First + 1;
        if(EntryA->YMin > EntryB->YMin)
        {
            edge_info Temp = *EntryA;
            *EntryA = *EntryB;
            *EntryB = Temp;
        }
    }
    else
    {
        u32 Half0 = Count / 2;
        u32 Half1 = Count - Half0;

        Assert(Half0 >= 1);
        Assert(Half1 >= 1);

        edge_info *InHalf0 = First;
        edge_info *InHalf1 = First + Half0;
        edge_info *End = First + Count;

        MergeSort(Half0, InHalf0, Temp);
        MergeSort(Half1, InHalf1, Temp);

        edge_info *ReadHalf0 = InHalf0;
        edge_info *ReadHalf1 = InHalf1;

        edge_info *Out = Temp;
        for(u32 Index = 0;
            Index < Count;
            ++Index)
        {
            if(ReadHalf0 == InHalf1)
            {
                *Out++ = *ReadHalf1++;
            }
            else if(ReadHalf1 == End)
            {
                *Out++ = *ReadHalf0++;
            }
            else if(ReadHalf0->YMin < ReadHalf1->YMin)
            {
                *Out++ = *ReadHalf0++;
            }
            else
            {
                *Out++ = *ReadHalf1++;
            }            
        }

        Assert(Out == (Temp + Count));
        Assert(ReadHalf0 == InHalf1);
        Assert(ReadHalf1 == End);
            
        for(u32 Index = 0;
            Index < Count;
            ++Index)
        {
            First[Index] = Temp[Index];
        }
    }
}

internal v3 
ProjectVertex(v3 CameraPosition, projective_transform *Transform)
{
    v3 Result = {};
    
    r32 DistanceAboveTarget = Transform->DistanceAboveTarget;
    
    r32 DistanceToPZ = (DistanceAboveTarget - CameraPosition.z);
    r32 NearClipPlane = 0.2f;
    
    v3 RawXY = V3(CameraPosition.xy, 1.0f);
    
    if(DistanceToPZ > NearClipPlane)
    {
        v3 ProjectedXY = (1.0f / DistanceToPZ)*Transform->FocalLength*RawXY;
        Result = V3(Transform->ScreenCenter, DistanceToPZ) + Transform->MetersToPixels*V3(ProjectedXY.xy, 0.0f);
    }

    return Result;
}

struct unproject_simd_result
{
    __m256 VertexX;
    __m256 VertexY;
    __m256 VertexZ;
};

inline unproject_simd_result
UnprojectVertex_8x(r32 X, r32 Y, __m256 CurrentZ_8x, projective_transform *Transform)
{
    unproject_simd_result Result = {};
    
    __m256 UnprojectedX_8x = {};
    __m256 UnprojectedY_8x = {};
    
    __m256 DistanceToPz_8x = _mm256_sub_ps(_mm256_set1_ps(Transform->DistanceAboveTarget), CurrentZ_8x);
    
    __m256 X_8x = _mm256_setr_ps(X + 0.0f,
                                X + 1.0f,
                                X + 2.0f,
                                X + 3.0f,
                                X + 4.0f,
                                X + 5.0f,
                                X + 6.0f,
                                X + 7.0f);
    
    __m256 Y_8x = _mm256_setr_ps(Y + 0.0f,
                                Y + 0.0f,
                                Y + 0.0f,
                                Y + 0.0f,
                                Y + 0.0f,
                                Y + 0.0f,
                                Y + 0.0f,
                                Y + 0.0f);
    
    __m256 InvMetersToPixels_8x = _mm256_set1_ps(1.0f/Transform->MetersToPixels);
    
    __m256 ScreenCenterX_8x = _mm256_set1_ps(Transform->ScreenCenter.x);
    __m256 ScreenCenterY_8x = _mm256_set1_ps(Transform->ScreenCenter.y);
    
    __m256 AX = _mm256_mul_ps(_mm256_sub_ps(X_8x, ScreenCenterX_8x), InvMetersToPixels_8x);
    __m256 AY = _mm256_mul_ps(_mm256_sub_ps(Y_8x, ScreenCenterY_8x), InvMetersToPixels_8x);
    
    __m256 FocalLength_8x = _mm256_set1_ps(Transform->FocalLength);
    
    Result.VertexX = _mm256_mul_ps(_mm256_div_ps(DistanceToPz_8x, FocalLength_8x), AX); 
    Result.VertexY = _mm256_mul_ps(_mm256_div_ps(DistanceToPz_8x, FocalLength_8x), AY); 
    Result.VertexZ = CurrentZ_8x;  
    
    return Result;
}

inline v3 
UnprojectVertex(v3 VertexP, projective_transform *Transform)
{
    v2 UnprojectedXY;

    r32 DistanceToPZ = Transform->DistanceAboveTarget - VertexP.z;
    
    v2 A = (VertexP.xy - Transform->ScreenCenter)*(1.0f/Transform->MetersToPixels);
    UnprojectedXY = ((DistanceToPZ)/Transform->FocalLength)*A;
    
    v3 Result = V3(UnprojectedXY, VertexP.z);
    
    return Result;
}

internal void
DrawModel(loaded_bitmap *Buffer,
             edge_info *Edges, u32 EdgeCount,
             game_render_commands *Commands,
             loaded_bitmap *Bitmap = 0,
             b32 PhongShading = 0)
{
    
    r32 *ZBuffer = Commands->ZBuffer;
    u32 ZBufferWidth = Commands->Width;
    
    s32 FirstRow = Edges[0].YMin;
    
    s32 Height;
    s32 MaxRow = Edges[0].YMax;
    for(u32 EdgeIndex = 1;
        EdgeIndex < EdgeCount;
        ++EdgeIndex)
    {
        if(MaxRow < Edges[EdgeIndex].YMax)
        {
            MaxRow = Edges[EdgeIndex].YMax;
        }
    }
    
    Height = MaxRow - FirstRow;

    edge_info *ListHead = 0;
    edge_info *ListTail = 0;
    
    s32 MaxY = FirstRow + Height;
    if(MaxY > Buffer->Height)
    {
        MaxY = Buffer->Height;
    }
    
    for(s32 RowIndex = FirstRow;
        RowIndex < MaxY;
        ++RowIndex)
    {
        for(u32 EdgeIndex = 0;
            EdgeIndex < EdgeCount;
            ++EdgeIndex)
        {
            edge_info *CurrentEdge = Edges + EdgeIndex;
            
            if(CurrentEdge->YMin == RowIndex)
            {
                if(ListHead)
                {
                    if(CurrentEdge->XMin < ListHead->XMin || 
                       (CurrentEdge->XMin == ListHead->XMin &&
                        (CurrentEdge->Gradient < ListHead->Gradient ||
                         (CurrentEdge->Gradient == ListHead->Gradient &&
                          CurrentEdge->Left < ListHead->Left))))
                    {
                        CurrentEdge->Next = ListHead;
                        ListHead = CurrentEdge;
                    }
                    else
                    {
                        edge_info *ComparedEdge = ListHead;
                        edge_info *PreviousEdge = ListHead;
                        while(ComparedEdge != ListTail)
                        {
                            ComparedEdge = ComparedEdge->Next;
                            
                            if(CurrentEdge->XMin < ComparedEdge->XMin || 
                               (CurrentEdge->XMin == ComparedEdge->XMin &&
                                (CurrentEdge->Gradient < ComparedEdge->Gradient ||
                                 (CurrentEdge->Gradient == ComparedEdge->Gradient &&
                                  CurrentEdge->Left < ComparedEdge->Left))))
                            {
                                CurrentEdge->Next = ComparedEdge;
                                PreviousEdge->Next = CurrentEdge;
                                ComparedEdge = ListTail;
                            }
                            else
                            {
                                PreviousEdge = ComparedEdge;
                            }
                            
                        }                        
                        
                        if(PreviousEdge == ComparedEdge)
                        {
                            ListTail->Next = CurrentEdge;
                            ListTail = CurrentEdge;
                        }
                    }

                }
                else
                {
                    ListHead = CurrentEdge;
                    ListTail = ListHead;
                }
            }
        }
        
        while(ListHead->YMax <= RowIndex)
        {
            edge_info *RemovedEdge = ListHead;
            ListHead = ListHead->Next;
            RemovedEdge->Next = 0;
        }
        
        edge_info *PreviousEdge = ListHead;
        edge_info *CheckedEdge = ListHead;
        
        if(CheckedEdge)
        {
            while(CheckedEdge != ListTail)
            {
                CheckedEdge = CheckedEdge->Next;
                
                if(CheckedEdge->YMax <= RowIndex)
                {
                    
                    if(CheckedEdge == ListTail)
                    {
                        ListTail = PreviousEdge;
                        ListTail->Next = 0;
                        CheckedEdge = ListTail;
                    }
                    else
                    {
                        PreviousEdge->Next = CheckedEdge->Next;
                        CheckedEdge = PreviousEdge;
                    }
                }
                
                PreviousEdge = CheckedEdge;
            }
        }
        
        edge_info *PreviousCurrentEdgeInList = 0;
        edge_info *PreviousNextEdgeInList = 0;
        edge_info *CurrentEdgeInList = ListHead;
        edge_info *NextEdgeInList = CurrentEdgeInList->Next;
        
        while(NextEdgeInList != 0)
        {
            
            r32 XOffset = 0.0f;
            
            if(RowIndex >= 0)
            {
            
                r32 XDifference = NextEdgeInList->XMin - CurrentEdgeInList->XMin;
                XDifference = roundf(XDifference);
                v4 LeftColor = CurrentEdgeInList->MinColor; 
                v4 RightColor = NextEdgeInList->MinColor; 
                
                v3 LeftNormal = CurrentEdgeInList->MinNormal;
                v3 RightNormal = NextEdgeInList->MinNormal;
                
                v2 LeftUV = V2(CurrentEdgeInList->UMin,
                               CurrentEdgeInList->VMin);
                
                v2 RightUV = V2(NextEdgeInList->UMin,
                               NextEdgeInList->VMin);
                
                r32 LeftOneOverZ = CurrentEdgeInList->OneOverZMin;
                r32 RightOneOverZ = NextEdgeInList->OneOverZMin;
                
                r32 OneOverZIncrement;
                v2 UVIncrement;
                v3 NormalIncrement;
                v4 ColorIncrement;
                r32 ZIncrement;
                if(XDifference != 0.0f)
                {
                    
                    OneOverZIncrement = (RightOneOverZ-LeftOneOverZ)/((r32)XDifference); 
                    
                    UVIncrement = 
                    {
                        (RightUV.u-LeftUV.u)/((r32)XDifference),
                        (RightUV.v-LeftUV.v)/((r32)XDifference),
                    };
                    
                    NormalIncrement = 
                    {
                        (RightNormal.x-LeftNormal.x)/((r32)XDifference),
                        (RightNormal.y-LeftNormal.y)/((r32)XDifference),
                        (RightNormal.z-LeftNormal.z)/((r32)XDifference),
                    };
                    
                    ColorIncrement = 
                    {
                        (RightColor.r-LeftColor.r)/((r32)XDifference),
                        (RightColor.g-LeftColor.g)/((r32)XDifference),
                        (RightColor.b-LeftColor.b)/((r32)XDifference),
                        (RightColor.a-LeftColor.a)/((r32)XDifference),
                    };
                    
                    ZIncrement = 
                    {
                        (NextEdgeInList->ZMin-CurrentEdgeInList->ZMin)/((r32)XDifference),
                    };
                }
                else
                {
                    OneOverZIncrement = 0.0f;
                    UVIncrement = {0.0f, 0.0f};
                    NormalIncrement = {0.0f, 0.0f, 0.0f};
                    ColorIncrement = {0.0f, 0.0f, 0.0f, 0.0f};
                    ZIncrement = 0.0f;
                }
                
                v4 CurrentColor;
                
                r32 CurrentZ = CurrentEdgeInList->ZMin;
                r32 CurrentOneOverZ = LeftOneOverZ;
                v2 CurrentUV = LeftUV;
                v3 CurrentNormal = LeftNormal;
                CurrentColor = LeftColor;   
                
                r32 LeftX = CurrentEdgeInList->XMin;
                if(LeftX < 0)
                {
                    XOffset = (r32)-LeftX;
                    LeftX = 0;
                }
                else if(LeftX >= Buffer->Width)
                {
                    LeftX = (r32)Buffer->Width - 1;
                }
                
                r32 RightX = NextEdgeInList->XMin;
                if(RightX < 0)
                {
                    RightX = 0;
                }
                else if(RightX >= Buffer->Width)
                {
                    RightX = (r32)Buffer->Width - 1;
                }
                
                LeftX = (r32)RoundR32ToS32(LeftX);
                RightX = (r32)RoundR32ToS32(RightX);
                
                s32 MinX = (s32)LeftX;
                s32 MaxX = (s32)RightX;
                
                CurrentZ += XOffset*ZIncrement;
                CurrentOneOverZ += XOffset*OneOverZIncrement;
                CurrentUV += XOffset*UVIncrement;
                CurrentNormal += XOffset*NormalIncrement;
                CurrentColor += XOffset*ColorIncrement;
                
                u8 *Row = (u8 *)Buffer->Memory +
                    MinX*BITMAP_BYTES_PER_PIXEL +
                    RowIndex*Buffer->Pitch;
                
                r32 *ZBufferPixel = (r32 *)ZBuffer +
                    MinX + RowIndex*ZBufferWidth;
                
                u32 *Pixel = (u32 *)Row;
                
                for(s32 X = MinX;
                    X <= MaxX;
                    ++X)
                {
                    if(Bitmap)
                    {
                        v2 FinalUV = (1.0f/CurrentOneOverZ)*CurrentUV;
                        v2 BitmapDim = V2i(Bitmap->Width-1, Bitmap->Height-1);
                        
                        v2 TexCoord = Hadamard(FinalUV, BitmapDim);
                        s32 TexelX = RoundR32ToS32(TexCoord.x);
                        s32 TexelY = RoundR32ToS32(TexCoord.y);
                        
                        u32 *Texel = (u32 *)((u8 *)Bitmap->Memory +
                                   TexelX*BITMAP_BYTES_PER_PIXEL +
                                   TexelY*Bitmap->Pitch);

                        r32 SA = (r32)((*Texel >> 24) & 0xFF)/255.0f;
                        r32 SR = (r32)((*Texel >> 16) & 0xFF)/255.0f;
                        r32 SG = (r32)((*Texel >> 8) & 0xFF)/255.0f;
                        r32 SB = (r32)((*Texel >> 0) & 0xFF)/255.0f;
                        
                        CurrentColor = V4(SR, SG, SB, SA);
                    }
                    
                    v4 FinalColor = {};
                    
                    if(PhongShading)
                    {
                        light_data *Lights = &Commands->LightData; 
                        light_info *Light = Lights->Lights;
                        
                        v3 CurrentVertexP = UnprojectVertex(V3((r32)X,
                                                               (r32)RowIndex,
                                                               CurrentZ),
                                                            &Commands->Transform);
                        
                        for(u32 LightIndex = 0;
                            LightIndex < Lights->LightCount;
                            ++LightIndex, ++Light)
                        {
                            if(LightIndex == 0)
                            {
                                FinalColor = Hadamard(CurrentColor, Lights->AmbientIntensity);
                            }
                            
                            v3 LightP = Light->P;
                            
                            v3 CurrentVectorToLight = Normalize(LightP - CurrentVertexP);
                            
                            v3 LightDirection = CurrentVectorToLight;
                            r32 CosineOfIncidence = Clamp01(Inner(CurrentNormal, CurrentVectorToLight));
                            v3 ViewDirection = Normalize(-CurrentVertexP);
                            v3 HalfAngle = Normalize(LightDirection + ViewDirection);
                            r32 PhongTerm = Clamp01(Inner(CurrentNormal, HalfAngle));
                            PhongTerm = pow(PhongTerm, 16);
                            
                            FinalColor += CosineOfIncidence*Hadamard(CurrentColor, Light->Intensity) + PhongTerm*Hadamard(V4(1.0f, 1.0f, 1.0f, 1.0f), Light->Intensity);
                        }
                        
                        FinalColor = Clamp01(FinalColor);
                        
                        r32 R = FinalColor.r;
                        r32 G = FinalColor.g;
                        r32 B = FinalColor.b;
                        r32 A = FinalColor.a;
                        
                        u32 Color32 = ((RoundR32ToU32(A * 255.0f) << 24) |
                                       (RoundR32ToU32(R * 255.0f) << 16) |
                                       (RoundR32ToU32(G * 255.0f) << 8) |
                                       (RoundR32ToU32(B * 255.0f) << 0));
                        
                        if(CurrentZ > *ZBufferPixel)
                        {
                            *ZBufferPixel = CurrentZ;
                            *Pixel = Color32;
                        }
                        
                        ZBufferPixel++;
                        Pixel++;
                        
                        CurrentNormal = Normalize(CurrentNormal + NormalIncrement);
                        CurrentColor = CurrentColor + ColorIncrement;
                        CurrentZ += ZIncrement;
                        CurrentOneOverZ += OneOverZIncrement;
                        CurrentUV += UVIncrement;
                    }
                    else
                    {
                        
                        FinalColor = CurrentColor;
                        
                        r32 R = FinalColor.r;
                        r32 G = FinalColor.g;
                        r32 B = FinalColor.b;
                        r32 A = FinalColor.a;
                        
                        u32 Color32 = ((RoundR32ToU32(A * 255.0f) << 24) |
                                       (RoundR32ToU32(R * 255.0f) << 16) |
                                       (RoundR32ToU32(G * 255.0f) << 8) |
                                       (RoundR32ToU32(B * 255.0f) << 0));
                        
                        if(CurrentZ > *ZBufferPixel)
                        {
                            *ZBufferPixel = CurrentZ;
                            *Pixel = Color32;
                        }
                        
                        ZBufferPixel++;
                        Pixel++;
                        
                        CurrentColor = CurrentColor + ColorIncrement;
                        CurrentZ += ZIncrement;
                        CurrentUV += UVIncrement;
                        CurrentOneOverZ += OneOverZIncrement;
                    }
                }
            }
            
            CurrentEdgeInList->XMin += CurrentEdgeInList->Gradient;
            NextEdgeInList->XMin += NextEdgeInList->Gradient;
            
            CurrentEdgeInList->ZMin += CurrentEdgeInList->ZGradient;
            NextEdgeInList->ZMin += NextEdgeInList->ZGradient;
            
            CurrentEdgeInList->MinColor += CurrentEdgeInList->ColorGradient;
            NextEdgeInList->MinColor += NextEdgeInList->ColorGradient;
            
            CurrentEdgeInList->MinNormal = Normalize(CurrentEdgeInList->MinNormal + CurrentEdgeInList->NormalGradient);
            NextEdgeInList->MinNormal = Normalize(NextEdgeInList->MinNormal + NextEdgeInList->NormalGradient);
            
            CurrentEdgeInList->UMin += CurrentEdgeInList->UGradient;
            CurrentEdgeInList->VMin += CurrentEdgeInList->VGradient;
            CurrentEdgeInList->OneOverZMin += CurrentEdgeInList->OneOverZGradient;

            NextEdgeInList->UMin += NextEdgeInList->UGradient;
            NextEdgeInList->VMin += NextEdgeInList->VGradient;
            NextEdgeInList->OneOverZMin += NextEdgeInList->OneOverZGradient;
            
            if(CurrentEdgeInList->XMin > NextEdgeInList->XMin)
            {
                CurrentEdgeInList->Next = NextEdgeInList->Next;
                NextEdgeInList->Next = CurrentEdgeInList;
                if(PreviousNextEdgeInList)
                {
                    PreviousNextEdgeInList->Next = NextEdgeInList;
                }
                CurrentEdgeInList = NextEdgeInList;
                NextEdgeInList = CurrentEdgeInList->Next;
            }
            
            if(PreviousNextEdgeInList)
            {
                if(PreviousNextEdgeInList->XMin > CurrentEdgeInList->XMin)
                {
                    PreviousNextEdgeInList->Next = CurrentEdgeInList->Next;
                    CurrentEdgeInList->Next = PreviousNextEdgeInList;
                    PreviousCurrentEdgeInList->Next = CurrentEdgeInList;
                    PreviousNextEdgeInList = CurrentEdgeInList;
                    CurrentEdgeInList = PreviousNextEdgeInList->Next;
                }
            }
            
            PreviousCurrentEdgeInList = CurrentEdgeInList;
            PreviousNextEdgeInList = NextEdgeInList;
            
            if(NextEdgeInList->Next)
            {
                CurrentEdgeInList = NextEdgeInList->Next;
                NextEdgeInList = CurrentEdgeInList->Next;
            }
            else
            {
                NextEdgeInList = 0;
            }
        }
    }
    
}

internal unproject_simd_result
NormalizeVector_8x(__m256 X, __m256 Y, __m256 Z)
{
    unproject_simd_result Result = {};
    
    
    
    __m256 VectorLength_8x = _mm256_sqrt_ps(_mm256_add_ps
                                            (_mm256_add_ps(_mm256_mul_ps(X, X),
                                                           _mm256_mul_ps(Y, Y)),
                                             _mm256_mul_ps(Z, Z)));
    
    Result.VertexX = _mm256_div_ps(X, VectorLength_8x);
    Result.VertexY = _mm256_div_ps(Y, VectorLength_8x);
    Result.VertexZ = _mm256_div_ps(Z, VectorLength_8x);
    
    return Result;
}

struct clip_mask_pair
{
    __m128i FirstHalf;
    __m128i SecondHalf;
    
};

internal void
FillLinesOptimized(game_render_commands *Commands,
                   loaded_bitmap *Buffer,
                   loaded_bitmap *Bitmap,
                   thread_edge_info *EdgesList,
                   u32 EdgeCount,
                   s32 RowIndex,
                   b32 PhongShading)
{
    thread_edge_info *CurrentPair = EdgesList;
    
    for(u32 PairIndex = 0;
        PairIndex < EdgeCount;
        ++PairIndex, ++CurrentPair)
    {
    
    edge_info CurrentEdgeInList_;
    edge_info NextEdgeInList_;
        
        CurrentEdgeInList_.XMin = CurrentPair->LeftXMin;
        NextEdgeInList_.XMin = CurrentPair->RightXMin;
        
        CurrentEdgeInList_.ZMin = CurrentPair->LeftZMin;
        NextEdgeInList_.ZMin = CurrentPair->RightZMin;
        
        CurrentEdgeInList_.OneOverZMin = CurrentPair->LeftOneOverZMin;
        NextEdgeInList_.OneOverZMin = CurrentPair->RightOneOverZMin;
        
        CurrentEdgeInList_.UMin = CurrentPair->LeftUMin;
        NextEdgeInList_.UMin = CurrentPair->RightUMin;
        
        CurrentEdgeInList_.VMin = CurrentPair->LeftVMin;
        NextEdgeInList_.VMin = CurrentPair->RightVMin;
        
        CurrentEdgeInList_.MinColor = CurrentPair->LeftMinColor;
        NextEdgeInList_.MinColor = CurrentPair->RightMinColor;
        
        CurrentEdgeInList_.MinNormal = CurrentPair->LeftMinNormal;
        NextEdgeInList_.MinNormal = CurrentPair->RightMinNormal;
        
        edge_info *CurrentEdgeInList = &CurrentEdgeInList_;
        edge_info *NextEdgeInList = &NextEdgeInList_;
        
        __m256 One_8x = _mm256_set1_ps(1.0f);
        __m256 One255_8x = _mm256_set1_ps(255.0f);
        __m256 Half_8x = _mm256_set1_ps(0.5f);
        __m256 Zero_8x = _mm256_set1_ps(0.0f);
        __m256i TexturePitch_8x = _mm256_set1_epi32(Bitmap->Pitch);
        __m256i MaskFF_8x = _mm256_set1_epi32(0xFF);
        __m256i MaskFFFF_8x = _mm256_set1_epi32(0xFFFF);      
        __m256i MaskFF00FF_8x = _mm256_set1_epi32(0x00FF00FF);
        
        r32 *ZBuffer = Commands->ZBuffer;
        u32 ZBufferWidth = Commands->Width;
        
        __m128i StartClipMask_4x = _mm_set1_epi8(-1);
        __m128i EndClipMask_4x = _mm_set1_epi8(-1);
        
        clip_mask_pair StartClipMaskPairs[] =
        {
            
            {_mm_slli_si128(StartClipMask_4x, 0), _mm_slli_si128(StartClipMask_4x, 0)},
            {_mm_slli_si128(StartClipMask_4x, 0), _mm_slli_si128(StartClipMask_4x, 4)},
            {_mm_slli_si128(StartClipMask_4x, 0), _mm_slli_si128(StartClipMask_4x, 8)},
            {_mm_slli_si128(StartClipMask_4x, 0), _mm_slli_si128(StartClipMask_4x, 12)},
            {_mm_slli_si128(StartClipMask_4x, 0), _mm_slli_si128(StartClipMask_4x, 16)},
            {_mm_slli_si128(StartClipMask_4x, 4), _mm_slli_si128(StartClipMask_4x, 16)},
            {_mm_slli_si128(StartClipMask_4x, 8), _mm_slli_si128(StartClipMask_4x, 16)},
            {_mm_slli_si128(StartClipMask_4x, 12), _mm_slli_si128(StartClipMask_4x, 16)},
            
        };
        
        clip_mask_pair EndClipMaskPairs[] =
        {
            {_mm_srli_si128(EndClipMask_4x, 16), _mm_srli_si128(EndClipMask_4x, 16)},
            {_mm_srli_si128(EndClipMask_4x, 16), _mm_srli_si128(EndClipMask_4x, 12)},
            {_mm_srli_si128(EndClipMask_4x, 16), _mm_srli_si128(EndClipMask_4x, 8)},
            {_mm_srli_si128(EndClipMask_4x, 16), _mm_srli_si128(EndClipMask_4x, 4)},
            {_mm_srli_si128(EndClipMask_4x, 16), _mm_srli_si128(EndClipMask_4x, 0)},
            {_mm_srli_si128(EndClipMask_4x, 12), _mm_srli_si128(EndClipMask_4x, 0)},
            {_mm_srli_si128(EndClipMask_4x, 8), _mm_srli_si128(EndClipMask_4x, 0)},
            {_mm_srli_si128(EndClipMask_4x, 4), _mm_srli_si128(EndClipMask_4x, 0)},
        };
        
        r32 XOffset = 0.0f;
        if(RowIndex >= 0)
        {
            
            r32 LeftX = CurrentEdgeInList->XMin;
            if(LeftX < 0)
            {
                XOffset = (r32)-LeftX;
                LeftX = 0;
            }
            else if(LeftX >= Buffer->Width)
            {
                LeftX = (r32)Buffer->Width - 1;
            }
            
            r32 RightX = NextEdgeInList->XMin;
            if(RightX < 0)
            {
                RightX = 0;
            }
            else if(RightX >= Buffer->Width)
            {
                RightX = (r32)Buffer->Width - 1;
            }
            
            s32 ColorLeftPosition = RoundR32ToS32(CurrentEdgeInList->XMin);
            s32 ColorRightPosition = RoundR32ToS32(NextEdgeInList->XMin);
            s32 XDifference = ColorRightPosition - ColorLeftPosition;
            
            v4 LeftColor = CurrentEdgeInList->MinColor; 
            v4 RightColor = NextEdgeInList->MinColor; 
            
            v3 LeftNormal = CurrentEdgeInList->MinNormal;
            v3 RightNormal = NextEdgeInList->MinNormal;
            
            v2 LeftUV = V2(CurrentEdgeInList->UMin,
                           CurrentEdgeInList->VMin);
            
            v2 RightUV = V2(NextEdgeInList->UMin,
                            NextEdgeInList->VMin);
            
            r32 LeftOneOverZ = CurrentEdgeInList->OneOverZMin;
            r32 RightOneOverZ = NextEdgeInList->OneOverZMin;
            
            
            LeftX = (r32)RoundR32ToS32(LeftX);
            RightX = (r32)RoundR32ToS32(RightX);
            
            s32 MinX = (s32)LeftX;
            s32 MaxX = (s32)RightX;
            
            __m256i StartClipMask_8x = _mm256_set1_epi32(-1);
            __m256i EndClipMask_8x = _mm256_set1_epi32(-1);
            
            if(MinX & 7)
            {
                clip_mask_pair Pair = StartClipMaskPairs[MinX & 7];
                StartClipMask_8x = _mm256_set_epi32(Pair.SecondHalf.m128i_u32[0],
                                                    Pair.SecondHalf.m128i_u32[1],
                                                    Pair.SecondHalf.m128i_u32[2],
                                                    Pair.SecondHalf.m128i_u32[3],
                                                    Pair.FirstHalf.m128i_u32[0],
                                                    Pair.FirstHalf.m128i_u32[1],
                                                    Pair.FirstHalf.m128i_u32[2],
                                                    Pair.FirstHalf.m128i_u32[3]);
                LeftX = (r32)(MinX & ~7);
                XOffset -= (r32)(MinX & 7)*1.0f;
            }
            
            if(MaxX & 7)
            {
                clip_mask_pair Pair = EndClipMaskPairs[MaxX & 7];
                EndClipMask_8x = _mm256_set_epi32(Pair.SecondHalf.m128i_u32[0],
                                                  Pair.SecondHalf.m128i_u32[1],
                                                  Pair.SecondHalf.m128i_u32[2],
                                                  Pair.SecondHalf.m128i_u32[3],
                                                  Pair.FirstHalf.m128i_u32[0],
                                                  Pair.FirstHalf.m128i_u32[1],
                                                  Pair.FirstHalf.m128i_u32[2],
                                                  Pair.FirstHalf.m128i_u32[3]);
                RightX = (r32)((MaxX & ~7) + 8);
            }
            
            
#if 1
            if((LeftX + 8) >= RightX)
            {
                
                __m128i StartClipMask_8xA = _mm_set_epi32(StartClipMask_8x.m256i_u32[0],
                                                          StartClipMask_8x.m256i_u32[1],
                                                          StartClipMask_8x.m256i_u32[2],
                                                          StartClipMask_8x.m256i_u32[3]);
                
                __m128i StartClipMask_8xB = _mm_set_epi32(StartClipMask_8x.m256i_u32[4],
                                                          StartClipMask_8x.m256i_u32[5],
                                                          StartClipMask_8x.m256i_u32[6],
                                                          StartClipMask_8x.m256i_u32[7]);
                
                __m128i EndClipMask_8xA = _mm_set_epi32(EndClipMask_8x.m256i_u32[0],
                                                        EndClipMask_8x.m256i_u32[1],
                                                        EndClipMask_8x.m256i_u32[2],
                                                        EndClipMask_8x.m256i_u32[3]);
                
                __m128i EndClipMask_8xB = _mm_set_epi32(EndClipMask_8x.m256i_u32[4],
                                                        EndClipMask_8x.m256i_u32[5],
                                                        EndClipMask_8x.m256i_u32[6],
                                                        EndClipMask_8x.m256i_u32[7]);
                
                StartClipMask_8xA = _mm_and_si128(StartClipMask_8xA, EndClipMask_8xA);
                StartClipMask_8xB = _mm_and_si128(StartClipMask_8xB, EndClipMask_8xB);
                
                StartClipMask_8x = _mm256_set_epi32(StartClipMask_8xB.m128i_u32[0],
                                                    StartClipMask_8xB.m128i_u32[1],
                                                    StartClipMask_8xB.m128i_u32[2],
                                                    StartClipMask_8xB.m128i_u32[3],
                                                    StartClipMask_8xA.m128i_u32[0],
                                                    StartClipMask_8xA.m128i_u32[1],
                                                    StartClipMask_8xA.m128i_u32[2],
                                                    StartClipMask_8xA.m128i_u32[3]);
                
            }
#endif
            
            r32 OneOverZIncrement = 0.0f;
            if(XDifference != 0)
            {
                OneOverZIncrement = (RightOneOverZ-LeftOneOverZ)/((r32)XDifference); 
            }
            
            __m256 CurrentOneOverZ_8x = _mm256_setr_ps(LeftOneOverZ + (XOffset + 0.0f)*OneOverZIncrement,
                                                       LeftOneOverZ + (XOffset + 1.0f)*OneOverZIncrement,
                                                       LeftOneOverZ + (XOffset + 2.0f)*OneOverZIncrement,
                                                       LeftOneOverZ + (XOffset + 3.0f)*OneOverZIncrement,
                                                       LeftOneOverZ + (XOffset + 4.0f)*OneOverZIncrement,
                                                       LeftOneOverZ + (XOffset + 5.0f)*OneOverZIncrement,
                                                       LeftOneOverZ + (XOffset + 6.0f)*OneOverZIncrement,
                                                       LeftOneOverZ + (XOffset + 7.0f)*OneOverZIncrement);
            
            
            __m256 OneOverZIncrement_8x = _mm256_set1_ps(OneOverZIncrement*8.0f);
            
            v2 UVIncrement = {};
            if(XDifference != 0)
            {
                UVIncrement = 
                {
                    (RightUV.u-LeftUV.u)/((r32)XDifference),
                    (RightUV.v-LeftUV.v)/((r32)XDifference),
                };
            }
            
            
            __m256 U_8x = _mm256_setr_ps(LeftUV.u + (XOffset + 0.0f)*UVIncrement.u,
                                         LeftUV.u + (XOffset + 1.0f)*UVIncrement.u,
                                         LeftUV.u + (XOffset + 2.0f)*UVIncrement.u,
                                         LeftUV.u + (XOffset + 3.0f)*UVIncrement.u,
                                         LeftUV.u + (XOffset + 4.0f)*UVIncrement.u,
                                         LeftUV.u + (XOffset + 5.0f)*UVIncrement.u,
                                         LeftUV.u + (XOffset + 6.0f)*UVIncrement.u,
                                         LeftUV.u + (XOffset + 7.0f)*UVIncrement.u);
            
            __m256 V_8x = _mm256_setr_ps(LeftUV.v + (XOffset + 0.0f)*UVIncrement.v,
                                         LeftUV.v + (XOffset + 1.0f)*UVIncrement.v,
                                         LeftUV.v + (XOffset + 2.0f)*UVIncrement.v,
                                         LeftUV.v + (XOffset + 3.0f)*UVIncrement.v,
                                         LeftUV.v + (XOffset + 4.0f)*UVIncrement.v,
                                         LeftUV.v + (XOffset + 5.0f)*UVIncrement.v,
                                         LeftUV.v + (XOffset + 6.0f)*UVIncrement.v,
                                         LeftUV.v + (XOffset + 7.0f)*UVIncrement.v);
            
            __m256 UIncrement_8x = _mm256_set1_ps(UVIncrement.u*8.0f);
            __m256 VIncrement_8x = _mm256_set1_ps(UVIncrement.v*8.0f);
            
            v3 NormalIncrement = {};
            if(XDifference != 0)
            {
                NormalIncrement = 
                {
                    (RightNormal.x-LeftNormal.x)/((r32)XDifference),
                    (RightNormal.y-LeftNormal.y)/((r32)XDifference),
                    (RightNormal.z-LeftNormal.z)/((r32)XDifference),
                };
            }
            
            __m256 NormalX_8x = _mm256_setr_ps(LeftNormal.x + (XOffset + 0.0f)*NormalIncrement.x,
                                               LeftNormal.x + (XOffset + 1.0f)*NormalIncrement.x,
                                               LeftNormal.x + (XOffset + 2.0f)*NormalIncrement.x,
                                               LeftNormal.x + (XOffset + 3.0f)*NormalIncrement.x,
                                               LeftNormal.x + (XOffset + 4.0f)*NormalIncrement.x,
                                               LeftNormal.x + (XOffset + 5.0f)*NormalIncrement.x,
                                               LeftNormal.x + (XOffset + 6.0f)*NormalIncrement.x,
                                               LeftNormal.x + (XOffset + 7.0f)*NormalIncrement.x);
            
            __m256 NormalY_8x = _mm256_setr_ps(LeftNormal.y + (XOffset + 0.0f)*NormalIncrement.y,
                                               LeftNormal.y + (XOffset + 1.0f)*NormalIncrement.y,
                                               LeftNormal.y + (XOffset + 2.0f)*NormalIncrement.y,
                                               LeftNormal.y + (XOffset + 3.0f)*NormalIncrement.y,
                                               LeftNormal.y + (XOffset + 4.0f)*NormalIncrement.y,
                                               LeftNormal.y + (XOffset + 5.0f)*NormalIncrement.y,
                                               LeftNormal.y + (XOffset + 6.0f)*NormalIncrement.y,
                                               LeftNormal.y + (XOffset + 7.0f)*NormalIncrement.y);
            
            __m256 NormalZ_8x = _mm256_setr_ps(LeftNormal.z + (XOffset + 0.0f)*NormalIncrement.z,
                                               LeftNormal.z + (XOffset + 1.0f)*NormalIncrement.z,
                                               LeftNormal.z + (XOffset + 2.0f)*NormalIncrement.z,
                                               LeftNormal.z + (XOffset + 3.0f)*NormalIncrement.z,
                                               LeftNormal.z + (XOffset + 4.0f)*NormalIncrement.z,
                                               LeftNormal.z + (XOffset + 5.0f)*NormalIncrement.z,
                                               LeftNormal.z + (XOffset + 6.0f)*NormalIncrement.z,
                                               LeftNormal.z + (XOffset + 7.0f)*NormalIncrement.z);
            
            unproject_simd_result CurrentNormal = NormalizeVector_8x(NormalX_8x, NormalY_8x, NormalZ_8x);
            
            NormalX_8x = CurrentNormal.VertexX;
            NormalY_8x = CurrentNormal.VertexY;
            NormalZ_8x = CurrentNormal.VertexZ;
            
            __m256 NormalXIncrement_8x = _mm256_set1_ps(NormalIncrement.x*8.0f);
            __m256 NormalYIncrement_8x = _mm256_set1_ps(NormalIncrement.y*8.0f);
            __m256 NormalZIncrement_8x = _mm256_set1_ps(NormalIncrement.z*8.0f);
            
            
            v4 ColorIncrement = {};
            if(XDifference != 0)
            {
                ColorIncrement = 
                {
                    (RightColor.r-LeftColor.r)/((r32)XDifference),
                    (RightColor.g-LeftColor.g)/((r32)XDifference),
                    (RightColor.b-LeftColor.b)/((r32)XDifference),
                    (RightColor.a-LeftColor.a)/((r32)XDifference),
                };
            }
            
            __m256 ColorR_8x = _mm256_set_ps(LeftColor.r + (XOffset + 0.0f)*ColorIncrement.r,
                                             LeftColor.r + (XOffset + 1.0f)*ColorIncrement.r,
                                             LeftColor.r + (XOffset + 2.0f)*ColorIncrement.r,
                                             LeftColor.r + (XOffset + 3.0f)*ColorIncrement.r,
                                             LeftColor.r + (XOffset + 4.0f)*ColorIncrement.r,
                                             LeftColor.r + (XOffset + 5.0f)*ColorIncrement.r,
                                             LeftColor.r + (XOffset + 6.0f)*ColorIncrement.r,
                                             LeftColor.r + (XOffset + 7.0f)*ColorIncrement.r);
            
            __m256 ColorG_8x = _mm256_set_ps(LeftColor.g + (XOffset + 0.0f)*ColorIncrement.g,
                                             LeftColor.g + (XOffset + 1.0f)*ColorIncrement.g,
                                             LeftColor.g + (XOffset + 2.0f)*ColorIncrement.g,
                                             LeftColor.g + (XOffset + 3.0f)*ColorIncrement.g,
                                             LeftColor.g + (XOffset + 4.0f)*ColorIncrement.g,
                                             LeftColor.g + (XOffset + 5.0f)*ColorIncrement.g,
                                             LeftColor.g + (XOffset + 6.0f)*ColorIncrement.g,
                                             LeftColor.g + (XOffset + 7.0f)*ColorIncrement.g);
            
            __m256 ColorB_8x = _mm256_set_ps(LeftColor.b + (XOffset + 0.0f)*ColorIncrement.b,
                                             LeftColor.b + (XOffset + 1.0f)*ColorIncrement.b,
                                             LeftColor.b + (XOffset + 2.0f)*ColorIncrement.b,
                                             LeftColor.b + (XOffset + 3.0f)*ColorIncrement.b,
                                             LeftColor.b + (XOffset + 4.0f)*ColorIncrement.b,
                                             LeftColor.b + (XOffset + 5.0f)*ColorIncrement.b,
                                             LeftColor.b + (XOffset + 6.0f)*ColorIncrement.b,
                                             LeftColor.b + (XOffset + 7.0f)*ColorIncrement.b);
            
            __m256 ColorA_8x = _mm256_set_ps(LeftColor.a + (XOffset + 0.0f)*ColorIncrement.a,
                                             LeftColor.a + (XOffset + 1.0f)*ColorIncrement.a,
                                             LeftColor.a + (XOffset + 2.0f)*ColorIncrement.a,
                                             LeftColor.a + (XOffset + 3.0f)*ColorIncrement.a,
                                             LeftColor.a + (XOffset + 4.0f)*ColorIncrement.a,
                                             LeftColor.a + (XOffset + 5.0f)*ColorIncrement.a,
                                             LeftColor.a + (XOffset + 6.0f)*ColorIncrement.a,
                                             LeftColor.a + (XOffset + 7.0f)*ColorIncrement.a);
            
            __m256 ColorRIncrement_8x = _mm256_set1_ps(ColorIncrement.r*8.0f);
            __m256 ColorGIncrement_8x = _mm256_set1_ps(ColorIncrement.g*8.0f);
            __m256 ColorBIncrement_8x = _mm256_set1_ps(ColorIncrement.b*8.0f);
            __m256 ColorAIncrement_8x = _mm256_set1_ps(ColorIncrement.a*8.0f);
            
            r32 CurrentZ = CurrentEdgeInList->ZMin;
            r32 ZIncrement = 0.0f;
            
            if(XDifference != 0)
            {
                ZIncrement = (NextEdgeInList->ZMin-CurrentEdgeInList->ZMin)/((r32)XDifference);
            }
            
            __m256 CurrentZ_8x = _mm256_setr_ps(CurrentEdgeInList->ZMin + (XOffset + 0.0f)*ZIncrement,
                                                CurrentEdgeInList->ZMin + (XOffset + 1.0f)*ZIncrement,
                                                CurrentEdgeInList->ZMin + (XOffset + 2.0f)*ZIncrement,
                                                CurrentEdgeInList->ZMin + (XOffset + 3.0f)*ZIncrement,
                                                CurrentEdgeInList->ZMin + (XOffset + 4.0f)*ZIncrement,
                                                CurrentEdgeInList->ZMin + (XOffset + 5.0f)*ZIncrement,
                                                CurrentEdgeInList->ZMin + (XOffset + 6.0f)*ZIncrement,
                                                CurrentEdgeInList->ZMin + (XOffset + 7.0f)*ZIncrement);
            
            __m256 ZIncrement_8x = _mm256_set1_ps(8.0f*ZIncrement);
            
            u8 *Row = (u8 *)Buffer->Memory +
                (s32)LeftX*BITMAP_BYTES_PER_PIXEL +
                RowIndex*Buffer->Pitch;
            
            r32 *ZBufferPixel = (r32 *)ZBuffer +
                (s32)LeftX + RowIndex*ZBufferWidth;
            
            u32 *Pixel = (u32 *)Row;
            __m256i ClipMask_8x = StartClipMask_8x;
            
            u8 *ZMaskPixel = Commands->ZMask + ((s32)LeftX + RowIndex*Commands->Width)/8;
            
            ClipMask_8x = _mm256_set_epi32(ClipMask_8x.m256i_u32[0],
                                           ClipMask_8x.m256i_u32[1],
                                           ClipMask_8x.m256i_u32[2],
                                           ClipMask_8x.m256i_u32[3],
                                           ClipMask_8x.m256i_u32[4],
                                           ClipMask_8x.m256i_u32[5],
                                           ClipMask_8x.m256i_u32[6],
                                           ClipMask_8x.m256i_u32[7]);
            
            for(s32 X = (s32)LeftX;
                X < (s32)RightX;
                X += 8)
            {
                
                void *TextureMemory = Bitmap->Memory;
                
                __m256 FinalU_8x = _mm256_mul_ps(_mm256_div_ps(One_8x, CurrentOneOverZ_8x), U_8x);
                __m256 FinalV_8x = _mm256_mul_ps(_mm256_div_ps(One_8x, CurrentOneOverZ_8x), V_8x);
                
                __m256 TexCoordX_8x = _mm256_set1_ps((r32)Bitmap->Width);
                __m256 TexCoordY_8x = _mm256_set1_ps((r32)Bitmap->Height);
                
                TexCoordX_8x = _mm256_mul_ps(TexCoordX_8x, FinalU_8x);
                TexCoordY_8x = _mm256_mul_ps(TexCoordY_8x, FinalV_8x);
                
                __m256i WriteMask = _mm256_castps_si256(_mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(FinalU_8x, Zero_8x, 13),
                                                                                    _mm256_cmp_ps(FinalU_8x, One_8x, 18)),
                                                                      _mm256_and_ps(_mm256_cmp_ps(FinalV_8x, Zero_8x, 13),
                                                                                    _mm256_cmp_ps(FinalV_8x, One_8x, 18))));
                
                WriteMask = _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(WriteMask), _mm256_castsi256_ps(ClipMask_8x))); 
                
                //__m256i FetchX_8x = _mm256_cvttps_epi32(TexCoordX_8x); 
                __m128i FetchX_8xA = _mm_cvttps_epi32(_mm_setr_ps(TexCoordX_8x.m256_f32[0],
                                                                  TexCoordX_8x.m256_f32[1],
                                                                  TexCoordX_8x.m256_f32[2],
                                                                  TexCoordX_8x.m256_f32[3]));
                
                __m128i FetchX_8xB = _mm_cvttps_epi32(_mm_setr_ps(TexCoordX_8x.m256_f32[4],
                                                                  TexCoordX_8x.m256_f32[5],
                                                                  TexCoordX_8x.m256_f32[6],
                                                                  TexCoordX_8x.m256_f32[7]));
                
                //FetchX_8x = _mm256_slli_epi32(FetchX_8x, 2);
                FetchX_8xA = _mm_slli_epi32(FetchX_8xA, 2);
                FetchX_8xB = _mm_slli_epi32(FetchX_8xB, 2);
                
                __m256i FetchY_8x = _mm256_cvttps_epi32(TexCoordY_8x); 
                
                
                //FetchY_8x = _mm256_or_si256(_mm256_mullo_epi16(FetchY_8x, TexturePitch_8x),                      
                //                            _mm256_slli_epi32(_mm256_mulhi_epi16(FetchY_8x, TexturePitch_8x), 16));
                
                __m128i FetchY_8xA = _mm_setr_epi32(FetchY_8x.m256i_i32[0],
                                                    FetchY_8x.m256i_i32[1],
                                                    FetchY_8x.m256i_i32[2],
                                                    FetchY_8x.m256i_i32[3]);
                
                __m128i FetchY_8xB = _mm_setr_epi32(FetchY_8x.m256i_i32[4],
                                                    FetchY_8x.m256i_i32[5],
                                                    FetchY_8x.m256i_i32[6],
                                                    FetchY_8x.m256i_i32[7]);
                
                __m128i TexturePitch_8xA = _mm_set1_epi32(TexturePitch_8x.m256i_u32[0]);
                
                __m128i TexturePitch_8xB = _mm_set1_epi32(TexturePitch_8x.m256i_u32[0]);
                
                FetchY_8xA = _mm_or_si128(_mm_mullo_epi16(FetchY_8xA, TexturePitch_8xA),                      
                                          _mm_slli_epi32(_mm_mulhi_epi16(FetchY_8xA, TexturePitch_8xA), 16));
                
                FetchY_8xB = _mm_or_si128(_mm_mullo_epi16(FetchY_8xB, TexturePitch_8xB),                      
                                          _mm_slli_epi32(_mm_mulhi_epi16(FetchY_8xB, TexturePitch_8xB), 16));
                
                //__m256i Fetch_8x = _mm256_add_epi32(FetchX_8x, FetchY_8x);
                __m128i Fetch_8xA = _mm_add_epi32(FetchX_8xA, FetchY_8xA);
                __m128i Fetch_8xB = _mm_add_epi32(FetchX_8xB, FetchY_8xB);
                
                __m256i Fetch_8x = _mm256_set_epi32(Fetch_8xA.m128i_i32[0],
                                                    Fetch_8xA.m128i_i32[1],
                                                    Fetch_8xA.m128i_i32[2],
                                                    Fetch_8xA.m128i_i32[3],
                                                    Fetch_8xB.m128i_i32[0],
                                                    Fetch_8xB.m128i_i32[1],
                                                    Fetch_8xB.m128i_i32[2],
                                                    Fetch_8xB.m128i_i32[3]);
                
                s32 Fetch0 = Fetch_8x.m256i_u32[0];
                s32 Fetch1 = Fetch_8x.m256i_u32[1];
                s32 Fetch2 = Fetch_8x.m256i_u32[2];
                s32 Fetch3 = Fetch_8x.m256i_u32[3];
                s32 Fetch4 = Fetch_8x.m256i_u32[4];
                s32 Fetch5 = Fetch_8x.m256i_u32[5];
                s32 Fetch6 = Fetch_8x.m256i_u32[6];
                s32 Fetch7 = Fetch_8x.m256i_u32[7];
                
                u8 *TexelPtr0 = ((u8 *)TextureMemory) + Fetch0;
                u8 *TexelPtr1 = ((u8 *)TextureMemory) + Fetch1;
                u8 *TexelPtr2 = ((u8 *)TextureMemory) + Fetch2;
                u8 *TexelPtr3 = ((u8 *)TextureMemory) + Fetch3;
                u8 *TexelPtr4 = ((u8 *)TextureMemory) + Fetch4;
                u8 *TexelPtr5 = ((u8 *)TextureMemory) + Fetch5;
                u8 *TexelPtr6 = ((u8 *)TextureMemory) + Fetch6;
                u8 *TexelPtr7 = ((u8 *)TextureMemory) + Fetch7;
                
#if 0
                __m256i Sample = _mm256_setr_epi32(*(u32 *)(TexelPtr0),
                                                   *(u32 *)(TexelPtr1),
                                                   *(u32 *)(TexelPtr2),
                                                   *(u32 *)(TexelPtr3),
                                                   *(u32 *)(TexelPtr4),
                                                   *(u32 *)(TexelPtr5),
                                                   *(u32 *)(TexelPtr6),
                                                   *(u32 *)(TexelPtr7));
#endif
                
                __m128i SampleA = _mm_setr_epi32(*(u32 *)(TexelPtr0),
                                                 *(u32 *)(TexelPtr1),
                                                 *(u32 *)(TexelPtr2),
                                                 *(u32 *)(TexelPtr3));
                
                __m128i SampleB = _mm_setr_epi32(*(u32 *)(TexelPtr4),
                                                 *(u32 *)(TexelPtr5),
                                                 *(u32 *)(TexelPtr6),
                                                 *(u32 *)(TexelPtr7));
                
                __m128i MaskFF_4x = _mm_set1_epi32(0xFF);
                
                __m128i SampleAR = _mm_and_si128(_mm_srli_epi32(SampleA, 24), MaskFF_4x);
                __m128i SampleBR = _mm_and_si128(_mm_srli_epi32(SampleB, 24), MaskFF_4x);
                
                __m128i SampleAG = _mm_and_si128(_mm_srli_epi32(SampleA, 16), MaskFF_4x);
                __m128i SampleBG = _mm_and_si128(_mm_srli_epi32(SampleB, 16), MaskFF_4x);
                
                __m128i SampleAB = _mm_and_si128(_mm_srli_epi32(SampleA, 8), MaskFF_4x);
                __m128i SampleBB = _mm_and_si128(_mm_srli_epi32(SampleB, 8), MaskFF_4x);
                
                __m128i SampleAA = _mm_and_si128(_mm_srli_epi32(SampleA, 0), MaskFF_4x);
                __m128i SampleBA = _mm_and_si128(_mm_srli_epi32(SampleB, 0), MaskFF_4x);
                
                __m256i SampleFinalR = _mm256_set_epi32(SampleAR.m128i_u32[0],
                                                        SampleAR.m128i_u32[1],
                                                        SampleAR.m128i_u32[2],
                                                        SampleAR.m128i_u32[3],
                                                        SampleBR.m128i_u32[0],
                                                        SampleBR.m128i_u32[1],
                                                        SampleBR.m128i_u32[2],
                                                        SampleBR.m128i_u32[3]);
                
                __m256i SampleFinalG = _mm256_set_epi32(SampleAG.m128i_u32[0],
                                                        SampleAG.m128i_u32[1],
                                                        SampleAG.m128i_u32[2],
                                                        SampleAG.m128i_u32[3],
                                                        SampleBG.m128i_u32[0],
                                                        SampleBG.m128i_u32[1],
                                                        SampleBG.m128i_u32[2],
                                                        SampleBG.m128i_u32[3]);
                
                __m256i SampleFinalB = _mm256_set_epi32(SampleAB.m128i_u32[0],
                                                        SampleAB.m128i_u32[1],
                                                        SampleAB.m128i_u32[2],
                                                        SampleAB.m128i_u32[3],
                                                        SampleBB.m128i_u32[0],
                                                        SampleBB.m128i_u32[1],
                                                        SampleBB.m128i_u32[2],
                                                        SampleBB.m128i_u32[3]);
                
                __m256i SampleFinalA = _mm256_set_epi32(SampleAA.m128i_u32[0],
                                                        SampleAA.m128i_u32[1],
                                                        SampleAA.m128i_u32[2],
                                                        SampleAA.m128i_u32[3],
                                                        SampleBA.m128i_u32[0],
                                                        SampleBA.m128i_u32[1],
                                                        SampleBA.m128i_u32[2],
                                                        SampleBA.m128i_u32[3]);
                
                //ColorA_8x = _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(Sample, 24), MaskFF_8x)), One255_8x);
                //ColorR_8x = _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(Sample, 16), MaskFF_8x)), One255_8x);
                //ColorG_8x = _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(Sample, 8), MaskFF_8x)), One255_8x);
                //ColorB_8x = _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(Sample, 0), MaskFF_8x)), One255_8x);
                
                ColorA_8x = _mm256_div_ps(_mm256_cvtepi32_ps(SampleFinalR), One255_8x);
                ColorR_8x = _mm256_div_ps(_mm256_cvtepi32_ps(SampleFinalG), One255_8x);
                ColorG_8x = _mm256_div_ps(_mm256_cvtepi32_ps(SampleFinalB), One255_8x);
                ColorB_8x = _mm256_div_ps(_mm256_cvtepi32_ps(SampleFinalA), One255_8x);
                
                
                __m256 FinalColorR_8x = _mm256_set1_ps(0.0f);
                __m256 FinalColorG_8x = _mm256_set1_ps(0.0f);
                __m256 FinalColorB_8x = _mm256_set1_ps(0.0f);
                __m256 FinalColorA_8x = _mm256_set1_ps(0.0f);
                
                if(PhongShading)
                {
                    light_data *Lights = &Commands->LightData; 
                    light_info *Light = Lights->Lights;
                    
                    unproject_simd_result CurrentVertexP = 
                        UnprojectVertex_8x((r32)X, (r32)RowIndex, CurrentZ_8x, &Commands->Transform);
                    
                    for(u32 LightIndex = 0;
                        LightIndex < Lights->LightCount;
                        ++LightIndex, ++Light)
                    {
                        if(LightIndex == 0)
                        {
                            FinalColorR_8x = _mm256_mul_ps(ColorR_8x, _mm256_set1_ps(Lights->AmbientIntensity.r));
                            FinalColorG_8x = _mm256_mul_ps(ColorG_8x, _mm256_set1_ps(Lights->AmbientIntensity.g));
                            FinalColorB_8x = _mm256_mul_ps(ColorB_8x, _mm256_set1_ps(Lights->AmbientIntensity.b));
                            FinalColorA_8x = _mm256_mul_ps(ColorA_8x, _mm256_set1_ps(Lights->AmbientIntensity.a));
                        }
                        
                        __m256 LightX = _mm256_set1_ps(Light->P.x);
                        __m256 LightY = _mm256_set1_ps(Light->P.y);
                        __m256 LightZ = _mm256_set1_ps(Light->P.z);
                        
                        __m256 VectorToLightX_8x = _mm256_sub_ps(LightX, CurrentVertexP.VertexX);
                        __m256 VectorToLightY_8x = _mm256_sub_ps(LightY, CurrentVertexP.VertexY);
                        __m256 VectorToLightZ_8x = _mm256_sub_ps(LightZ, CurrentVertexP.VertexZ);
                        
                        unproject_simd_result VectorToLight = NormalizeVector_8x(VectorToLightX_8x, VectorToLightY_8x, VectorToLightZ_8x);
                        
                        unproject_simd_result LightDirection = VectorToLight;
                        
                        __m256 DotNormalLightX_8x = _mm256_mul_ps(NormalX_8x, VectorToLight.VertexX);
                        __m256 DotNormalLightY_8x = _mm256_mul_ps(NormalY_8x, VectorToLight.VertexY);
                        __m256 DotNormalLightZ_8x = _mm256_mul_ps(NormalZ_8x, VectorToLight.VertexZ);
                        
                        __m256 CosineOfIncidence_8x = _mm256_min_ps(One_8x, _mm256_max_ps(Zero_8x, _mm256_add_ps(_mm256_add_ps(DotNormalLightX_8x, DotNormalLightY_8x), DotNormalLightZ_8x)));
                        
                        unproject_simd_result ViewDirection = NormalizeVector_8x(
                            _mm256_sub_ps(Zero_8x, CurrentVertexP.VertexX), 
                            _mm256_sub_ps(Zero_8x, CurrentVertexP.VertexY), 
                            _mm256_sub_ps(Zero_8x, CurrentVertexP.VertexZ));
                        
                        __m256 HalfAngleX_8x = _mm256_add_ps(LightDirection.VertexX, ViewDirection.VertexX);
                        __m256 HalfAngleY_8x = _mm256_add_ps(LightDirection.VertexY, ViewDirection.VertexY);
                        __m256 HalfAngleZ_8x = _mm256_add_ps(LightDirection.VertexZ, ViewDirection.VertexZ);
                        
                        unproject_simd_result HalfAngle = NormalizeVector_8x(HalfAngleX_8x, HalfAngleY_8x, HalfAngleZ_8x);
                        
                        __m256 PhongTerm_8x = _mm256_min_ps(One_8x, _mm256_max_ps(Zero_8x, _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(NormalX_8x, HalfAngle.VertexX), _mm256_mul_ps(NormalY_8x, HalfAngle.VertexY)), _mm256_mul_ps(NormalZ_8x, HalfAngle.VertexZ))));
                        
                        for(u32 FactorIndex = 0;
                            FactorIndex < 4;
                            ++FactorIndex)
                        {
                            PhongTerm_8x = _mm256_mul_ps(PhongTerm_8x, PhongTerm_8x);
                        }
                        
                        __m256 SpecularColorR_8x = _mm256_set1_ps(1.0f); 
                        __m256 SpecularColorG_8x = _mm256_set1_ps(1.0f); 
                        __m256 SpecularColorB_8x = _mm256_set1_ps(1.0f); 
                        __m256 SpecularColorA_8x = _mm256_set1_ps(1.0f); 
                        
                        __m256 LightIntensityR_8x = _mm256_set1_ps(Light->Intensity.r); 
                        __m256 LightIntensityG_8x = _mm256_set1_ps(Light->Intensity.g); 
                        __m256 LightIntensityB_8x = _mm256_set1_ps(Light->Intensity.b); 
                        __m256 LightIntensityA_8x = _mm256_set1_ps(Light->Intensity.a); 
                        
                        FinalColorR_8x = _mm256_add_ps(FinalColorR_8x, 
                                                       _mm256_add_ps(_mm256_mul_ps(CosineOfIncidence_8x, _mm256_mul_ps(ColorR_8x, LightIntensityR_8x)), 
                                                                     _mm256_mul_ps(PhongTerm_8x, _mm256_mul_ps(SpecularColorR_8x, LightIntensityR_8x))));
                        
                        FinalColorG_8x = _mm256_add_ps(FinalColorG_8x, 
                                                       _mm256_add_ps(_mm256_mul_ps(CosineOfIncidence_8x, _mm256_mul_ps(ColorG_8x, LightIntensityG_8x)), 
                                                                     _mm256_mul_ps(PhongTerm_8x, _mm256_mul_ps(SpecularColorG_8x, LightIntensityG_8x))));
                        
                        FinalColorB_8x = _mm256_add_ps(FinalColorB_8x, 
                                                       _mm256_add_ps(_mm256_mul_ps(CosineOfIncidence_8x, _mm256_mul_ps(ColorB_8x, LightIntensityB_8x)), 
                                                                     _mm256_mul_ps(PhongTerm_8x, _mm256_mul_ps(SpecularColorB_8x, LightIntensityB_8x))));
                        
                        FinalColorA_8x = _mm256_add_ps(FinalColorA_8x, 
                                                       _mm256_add_ps(_mm256_mul_ps(CosineOfIncidence_8x, _mm256_mul_ps(ColorA_8x, LightIntensityA_8x)),
                                                                     _mm256_mul_ps(PhongTerm_8x, _mm256_mul_ps(SpecularColorA_8x, LightIntensityA_8x))));
                    }
                    
                    FinalColorR_8x = _mm256_max_ps(_mm256_min_ps(FinalColorR_8x, One_8x), Zero_8x);
                    FinalColorG_8x = _mm256_max_ps(_mm256_min_ps(FinalColorG_8x, One_8x), Zero_8x);
                    FinalColorB_8x = _mm256_max_ps(_mm256_min_ps(FinalColorB_8x, One_8x), Zero_8x);
                    FinalColorA_8x = _mm256_max_ps(_mm256_min_ps(FinalColorA_8x, One_8x), Zero_8x);
                    
                    //__m256i Color32_8x = _mm256_or_si256(_mm256_or_si256(_mm256_or_si256(_mm256_srli_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(FinalColorG_8x, One255_8x)), 8), 
                    //                                                                     _mm256_cvtps_epi32(_mm256_mul_ps(FinalColorG_8x, One255_8x))),
                    //                                                     _mm256_srli_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(FinalColorR_8x, One255_8x)), 16)),
                    //                                     _mm256_srli_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(FinalColorA_8x, One255_8x)), 24));
                    
                    __m256i Color32R_8x = _mm256_cvtps_epi32(_mm256_mul_ps(FinalColorR_8x, One255_8x));
                    __m256i Color32G_8x = _mm256_cvtps_epi32(_mm256_mul_ps(FinalColorG_8x, One255_8x));
                    __m256i Color32B_8x = _mm256_cvtps_epi32(_mm256_mul_ps(FinalColorB_8x, One255_8x));
                    __m256i Color32A_8x = _mm256_cvtps_epi32(_mm256_mul_ps(FinalColorA_8x, One255_8x));
                    
                    __m128i Color32R_8xA = _mm_set_epi32(Color32R_8x.m256i_u32[0],
                                                         Color32R_8x.m256i_u32[1],
                                                         Color32R_8x.m256i_u32[2],
                                                         Color32R_8x.m256i_u32[3]);
                    
                    __m128i Color32R_8xB = _mm_set_epi32(Color32R_8x.m256i_u32[4],
                                                         Color32R_8x.m256i_u32[5],
                                                         Color32R_8x.m256i_u32[6],
                                                         Color32R_8x.m256i_u32[7]);
                    
                    __m128i Color32G_8xA = _mm_set_epi32(Color32G_8x.m256i_u32[0],
                                                         Color32G_8x.m256i_u32[1],
                                                         Color32G_8x.m256i_u32[2],
                                                         Color32G_8x.m256i_u32[3]);
                    
                    __m128i Color32G_8xB = _mm_set_epi32(Color32G_8x.m256i_u32[4],
                                                         Color32G_8x.m256i_u32[5],
                                                         Color32G_8x.m256i_u32[6],
                                                         Color32G_8x.m256i_u32[7]);
                    
                    __m128i Color32B_8xA = _mm_set_epi32(Color32B_8x.m256i_u32[0],
                                                         Color32B_8x.m256i_u32[1],
                                                         Color32B_8x.m256i_u32[2],
                                                         Color32B_8x.m256i_u32[3]);
                    
                    __m128i Color32B_8xB = _mm_set_epi32(Color32B_8x.m256i_u32[4],
                                                         Color32B_8x.m256i_u32[5],
                                                         Color32B_8x.m256i_u32[6],
                                                         Color32B_8x.m256i_u32[7]);
                    
                    __m128i Color32A_8xA = _mm_set_epi32(Color32A_8x.m256i_u32[0],
                                                         Color32A_8x.m256i_u32[1],
                                                         Color32A_8x.m256i_u32[2],
                                                         Color32A_8x.m256i_u32[3]);
                    
                    __m128i Color32A_8xB = _mm_set_epi32(Color32A_8x.m256i_u32[4],
                                                         Color32A_8x.m256i_u32[5],
                                                         Color32A_8x.m256i_u32[6],
                                                         Color32A_8x.m256i_u32[7]);
                    
                    Color32R_8xA = _mm_slli_epi32(Color32R_8xA, 16);
                    Color32R_8xB = _mm_slli_epi32(Color32R_8xB, 16);
                    Color32G_8xA = _mm_slli_epi32(Color32G_8xA, 8);
                    Color32G_8xB = _mm_slli_epi32(Color32G_8xB, 8);
                    Color32B_8xA = _mm_slli_epi32(Color32B_8xA, 0);
                    Color32B_8xB = _mm_slli_epi32(Color32B_8xB, 0);
                    Color32A_8xA = _mm_slli_epi32(Color32A_8xA, 24);
                    Color32A_8xB = _mm_slli_epi32(Color32A_8xB, 24);
                    
                    __m128i Color32_8xA = _mm_or_si128(_mm_or_si128(_mm_or_si128(Color32R_8xA, Color32G_8xA), Color32B_8xA), Color32A_8xA);
                    __m128i Color32_8xB = _mm_or_si128(_mm_or_si128(_mm_or_si128(Color32R_8xB, Color32G_8xB), Color32B_8xB), Color32A_8xB);
                    
                    __m256i Color32_8x = _mm256_set_epi32(Color32_8xB.m128i_u32[0],
                                                          Color32_8xB.m128i_u32[1],
                                                          Color32_8xB.m128i_u32[2],
                                                          Color32_8xB.m128i_u32[3],
                                                          Color32_8xA.m128i_u32[0],
                                                          Color32_8xA.m128i_u32[1],
                                                          Color32_8xA.m128i_u32[2],
                                                          Color32_8xA.m128i_u32[3]
                                                          );
                    
                    u8 volatile FinalMask = 1;
                    
                    __m256i FinalWriteMask = WriteMask;
                    __m256i FinalColor32_8x = Color32_8x;
                    __m256i DebugColor32_8x = _mm256_set1_epi8(-1);
                    
                    while(FinalMask != 0)
                    {
                        
                        FinalMask = _InterlockedCompareExchange8((char volatile *)ZMaskPixel,
                                                                 1,
                                                                 0);
                        
                        if(FinalMask == 0)
                        {
                            
                            __m256 OriginalZ_8x = _mm256_load_ps(ZBufferPixel); 
                            __m256 ZMask = _mm256_cmp_ps(CurrentZ_8x, OriginalZ_8x, 30);
                            
                            ZMask = _mm256_and_ps(ZMask, _mm256_castsi256_ps(WriteMask));
                            
                            FinalWriteMask = _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(WriteMask), ZMask));
                            
                            __m256 NewZ_8x = _mm256_or_ps(_mm256_and_ps(ZMask, CurrentZ_8x),
                                                          _mm256_andnot_ps(ZMask, OriginalZ_8x));
                            
                            __m256i OriginalDest = _mm256_loadu_si256((__m256i *)Pixel);
                            
                            FinalColor32_8x = _mm256_castps_si256(_mm256_or_ps(_mm256_and_ps(_mm256_castsi256_ps(FinalWriteMask), _mm256_castsi256_ps(FinalColor32_8x)),
                                                                               _mm256_andnot_ps(_mm256_castsi256_ps(FinalWriteMask), _mm256_castsi256_ps(OriginalDest))));
                            
                            _mm256_storeu_ps(ZBufferPixel, NewZ_8x);
                            _mm256_storeu_si256((__m256i *)Pixel, FinalColor32_8x);
                            _WriteBarrier();
                            
                            *ZMaskPixel = 0;
                        }
                    }
                    
                    if((X + 16) < RightX)
                    {                  
                        ClipMask_8x = _mm256_set1_epi8(-1);
                    }
                    else
                    {                  
                        ClipMask_8x = EndClipMask_8x;
                        ClipMask_8x = _mm256_set_epi32(ClipMask_8x.m256i_u32[0],
                                                       ClipMask_8x.m256i_u32[1],
                                                       ClipMask_8x.m256i_u32[2],
                                                       ClipMask_8x.m256i_u32[3],
                                                       ClipMask_8x.m256i_u32[4],
                                                       ClipMask_8x.m256i_u32[5],
                                                       ClipMask_8x.m256i_u32[6],
                                                       ClipMask_8x.m256i_u32[7]);
                    }
                    
                    ZBufferPixel += 8;
                    Pixel += 8;
                    ZMaskPixel++;
                    
                    unproject_simd_result NewNormals = NormalizeVector_8x(
                        _mm256_add_ps(NormalX_8x, NormalXIncrement_8x),
                        _mm256_add_ps(NormalY_8x, NormalYIncrement_8x),
                        _mm256_add_ps(NormalZ_8x, NormalZIncrement_8x)
                        );
                    
                    NormalX_8x = NewNormals.VertexX;
                    NormalY_8x = NewNormals.VertexY;
                    NormalZ_8x = NewNormals.VertexZ;
                    
                    ColorR_8x = _mm256_add_ps(ColorR_8x, ColorRIncrement_8x);
                    ColorG_8x = _mm256_add_ps(ColorG_8x, ColorGIncrement_8x);
                    ColorB_8x = _mm256_add_ps(ColorB_8x, ColorBIncrement_8x);
                    ColorA_8x = _mm256_add_ps(ColorA_8x, ColorAIncrement_8x);
                    
                    CurrentZ_8x = _mm256_add_ps(CurrentZ_8x, ZIncrement_8x);
                    
                    CurrentOneOverZ_8x = _mm256_add_ps(CurrentOneOverZ_8x, OneOverZIncrement_8x);
                    
                    U_8x = _mm256_add_ps(U_8x, UIncrement_8x);
                    V_8x = _mm256_add_ps(V_8x, VIncrement_8x);
                    
                }
                else
                {
                    
                    
                    FinalColorR_8x = ColorR_8x;
                    FinalColorR_8x = ColorG_8x;
                    FinalColorR_8x = ColorB_8x;
                    FinalColorR_8x = ColorA_8x;
                    
                    __m256i Color32_8x = _mm256_or_si256(_mm256_or_si256(_mm256_or_si256(_mm256_srli_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(FinalColorG_8x, One255_8x)), 8), 
                                                                                         _mm256_cvtps_epi32(_mm256_mul_ps(FinalColorG_8x, One255_8x))),
                                                                         _mm256_srli_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(FinalColorR_8x, One255_8x)), 16)),
                                                         _mm256_srli_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(FinalColorA_8x, One255_8x)), 24));
                    
                    if(CurrentZ > *ZBufferPixel)
                    {
                        _mm256_storeu_ps(ZBufferPixel, CurrentZ_8x);
                        _mm256_storeu_si256((__m256i *)Pixel, Color32_8x);
                    }
                    ZBufferPixel += 8;
                    Pixel += 8;
                    
                    
                    
                    ColorR_8x = _mm256_add_ps(ColorR_8x, ColorRIncrement_8x);
                    ColorG_8x = _mm256_add_ps(ColorG_8x, ColorGIncrement_8x);
                    ColorB_8x = _mm256_add_ps(ColorB_8x, ColorBIncrement_8x);
                    ColorA_8x = _mm256_add_ps(ColorA_8x, ColorAIncrement_8x);
                    
                    CurrentZ_8x = _mm256_add_ps(CurrentZ_8x, ZIncrement_8x);
                    
                }
            }
        }
    }    
}

internal void
FillLineOptimized(game_render_commands *Commands,
                  loaded_bitmap *Buffer,
                  loaded_bitmap *Bitmap,
                  edge_info *CurrentEdgeInList, 
                  edge_info *NextEdgeInList, 
                  s32 RowIndex,
                  b32 PhongShading)
{
    
    __m256 One_8x = _mm256_set1_ps(1.0f);
    __m256 One255_8x = _mm256_set1_ps(255.0f);
    __m256 Half_8x = _mm256_set1_ps(0.5f);
    __m256 Zero_8x = _mm256_set1_ps(0.0f);
    __m256i TexturePitch_8x = _mm256_set1_epi32(Bitmap->Pitch);
    __m256i MaskFF_8x = _mm256_set1_epi32(0xFF);
    __m256i MaskFFFF_8x = _mm256_set1_epi32(0xFFFF);      
    __m256i MaskFF00FF_8x = _mm256_set1_epi32(0x00FF00FF);
    
    r32 *ZBuffer = Commands->ZBuffer;
    u32 ZBufferWidth = Commands->Width;
    
    __m128i StartClipMask_4x = _mm_set1_epi8(-1);
    __m128i EndClipMask_4x = _mm_set1_epi8(-1);
    
    clip_mask_pair StartClipMaskPairs[] =
    {
        
        {_mm_slli_si128(StartClipMask_4x, 0), _mm_slli_si128(StartClipMask_4x, 0)},
        {_mm_slli_si128(StartClipMask_4x, 0), _mm_slli_si128(StartClipMask_4x, 4)},
        {_mm_slli_si128(StartClipMask_4x, 0), _mm_slli_si128(StartClipMask_4x, 8)},
        {_mm_slli_si128(StartClipMask_4x, 0), _mm_slli_si128(StartClipMask_4x, 12)},
        {_mm_slli_si128(StartClipMask_4x, 0), _mm_slli_si128(StartClipMask_4x, 16)},
        {_mm_slli_si128(StartClipMask_4x, 4), _mm_slli_si128(StartClipMask_4x, 16)},
        {_mm_slli_si128(StartClipMask_4x, 8), _mm_slli_si128(StartClipMask_4x, 16)},
        {_mm_slli_si128(StartClipMask_4x, 12), _mm_slli_si128(StartClipMask_4x, 16)},
        
    };
    
    clip_mask_pair EndClipMaskPairs[] =
    {
        {_mm_srli_si128(EndClipMask_4x, 16), _mm_srli_si128(EndClipMask_4x, 16)},
        {_mm_srli_si128(EndClipMask_4x, 16), _mm_srli_si128(EndClipMask_4x, 12)},
        {_mm_srli_si128(EndClipMask_4x, 16), _mm_srli_si128(EndClipMask_4x, 8)},
        {_mm_srli_si128(EndClipMask_4x, 16), _mm_srli_si128(EndClipMask_4x, 4)},
        {_mm_srli_si128(EndClipMask_4x, 16), _mm_srli_si128(EndClipMask_4x, 0)},
        {_mm_srli_si128(EndClipMask_4x, 12), _mm_srli_si128(EndClipMask_4x, 0)},
        {_mm_srli_si128(EndClipMask_4x, 8), _mm_srli_si128(EndClipMask_4x, 0)},
        {_mm_srli_si128(EndClipMask_4x, 4), _mm_srli_si128(EndClipMask_4x, 0)},
    };
    
    r32 XOffset = 0.0f;
    if(RowIndex >= 0)
    {
        
        r32 LeftX = CurrentEdgeInList->XMin;
        if(LeftX < 0)
        {
            XOffset = (r32)-LeftX;
            LeftX = 0;
        }
        else if(LeftX >= Buffer->Width)
        {
            LeftX = (r32)Buffer->Width - 1;
        }
        
        r32 RightX = NextEdgeInList->XMin;
        if(RightX < 0)
        {
            RightX = 0;
        }
        else if(RightX >= Buffer->Width)
        {
            RightX = (r32)Buffer->Width - 1;
        }
        
        s32 ColorLeftPosition = RoundR32ToS32(CurrentEdgeInList->XMin);
        s32 ColorRightPosition = RoundR32ToS32(NextEdgeInList->XMin);
        s32 XDifference = ColorRightPosition - ColorLeftPosition;
        
        v4 LeftColor = CurrentEdgeInList->MinColor; 
        v4 RightColor = NextEdgeInList->MinColor; 
        
        v3 LeftNormal = CurrentEdgeInList->MinNormal;
        v3 RightNormal = NextEdgeInList->MinNormal;
        
        v2 LeftUV = V2(CurrentEdgeInList->UMin,
                       CurrentEdgeInList->VMin);
        
        v2 RightUV = V2(NextEdgeInList->UMin,
                        NextEdgeInList->VMin);
        
        r32 LeftOneOverZ = CurrentEdgeInList->OneOverZMin;
        r32 RightOneOverZ = NextEdgeInList->OneOverZMin;
        
        
        LeftX = (r32)RoundR32ToS32(LeftX);
        RightX = (r32)RoundR32ToS32(RightX);
        
        s32 MinX = (s32)LeftX;
        s32 MaxX = (s32)RightX;
        
        __m256i StartClipMask_8x = _mm256_set1_epi32(-1);
        __m256i EndClipMask_8x = _mm256_set1_epi32(-1);
        
        if(MinX & 7)
        {
            clip_mask_pair Pair = StartClipMaskPairs[MinX & 7];
            StartClipMask_8x = _mm256_set_epi32(Pair.SecondHalf.m128i_u32[0],
                                                Pair.SecondHalf.m128i_u32[1],
                                                Pair.SecondHalf.m128i_u32[2],
                                                Pair.SecondHalf.m128i_u32[3],
                                                Pair.FirstHalf.m128i_u32[0],
                                                Pair.FirstHalf.m128i_u32[1],
                                                Pair.FirstHalf.m128i_u32[2],
                                                Pair.FirstHalf.m128i_u32[3]);
            LeftX = (r32)(MinX & ~7);
            XOffset -= (r32)(MinX & 7)*1.0f;
        }
        
        if(MaxX & 7)
        {
            clip_mask_pair Pair = EndClipMaskPairs[MaxX & 7];
            EndClipMask_8x = _mm256_set_epi32(Pair.SecondHalf.m128i_u32[0],
                                              Pair.SecondHalf.m128i_u32[1],
                                              Pair.SecondHalf.m128i_u32[2],
                                              Pair.SecondHalf.m128i_u32[3],
                                              Pair.FirstHalf.m128i_u32[0],
                                              Pair.FirstHalf.m128i_u32[1],
                                              Pair.FirstHalf.m128i_u32[2],
                                              Pair.FirstHalf.m128i_u32[3]);
            RightX = (r32)((MaxX & ~7) + 8);
        }
        
        
#if 1
        if((LeftX + 8) >= RightX)
        {
            
            __m128i StartClipMask_8xA = _mm_set_epi32(StartClipMask_8x.m256i_u32[0],
                                                      StartClipMask_8x.m256i_u32[1],
                                                      StartClipMask_8x.m256i_u32[2],
                                                      StartClipMask_8x.m256i_u32[3]);
            
            __m128i StartClipMask_8xB = _mm_set_epi32(StartClipMask_8x.m256i_u32[4],
                                                      StartClipMask_8x.m256i_u32[5],
                                                      StartClipMask_8x.m256i_u32[6],
                                                      StartClipMask_8x.m256i_u32[7]);
            
            __m128i EndClipMask_8xA = _mm_set_epi32(EndClipMask_8x.m256i_u32[0],
                                                    EndClipMask_8x.m256i_u32[1],
                                                    EndClipMask_8x.m256i_u32[2],
                                                    EndClipMask_8x.m256i_u32[3]);
            
            __m128i EndClipMask_8xB = _mm_set_epi32(EndClipMask_8x.m256i_u32[4],
                                                    EndClipMask_8x.m256i_u32[5],
                                                    EndClipMask_8x.m256i_u32[6],
                                                    EndClipMask_8x.m256i_u32[7]);
            
            StartClipMask_8xA = _mm_and_si128(StartClipMask_8xA, EndClipMask_8xA);
            StartClipMask_8xB = _mm_and_si128(StartClipMask_8xB, EndClipMask_8xB);
            
            StartClipMask_8x = _mm256_set_epi32(StartClipMask_8xB.m128i_u32[0],
                                                StartClipMask_8xB.m128i_u32[1],
                                                StartClipMask_8xB.m128i_u32[2],
                                                StartClipMask_8xB.m128i_u32[3],
                                                StartClipMask_8xA.m128i_u32[0],
                                                StartClipMask_8xA.m128i_u32[1],
                                                StartClipMask_8xA.m128i_u32[2],
                                                StartClipMask_8xA.m128i_u32[3]);
            
        }
#endif
        
        r32 OneOverZIncrement = 0.0f;
        if(XDifference != 0)
        {
            OneOverZIncrement = (RightOneOverZ-LeftOneOverZ)/((r32)XDifference); 
        }
        
        __m256 CurrentOneOverZ_8x = _mm256_setr_ps(LeftOneOverZ + (XOffset + 0.0f)*OneOverZIncrement,
                                                   LeftOneOverZ + (XOffset + 1.0f)*OneOverZIncrement,
                                                   LeftOneOverZ + (XOffset + 2.0f)*OneOverZIncrement,
                                                   LeftOneOverZ + (XOffset + 3.0f)*OneOverZIncrement,
                                                   LeftOneOverZ + (XOffset + 4.0f)*OneOverZIncrement,
                                                   LeftOneOverZ + (XOffset + 5.0f)*OneOverZIncrement,
                                                   LeftOneOverZ + (XOffset + 6.0f)*OneOverZIncrement,
                                                   LeftOneOverZ + (XOffset + 7.0f)*OneOverZIncrement);
        
        
        __m256 OneOverZIncrement_8x = _mm256_set1_ps(OneOverZIncrement*8.0f);
        
        v2 UVIncrement = {};
        if(XDifference != 0)
        {
            UVIncrement = 
            {
                (RightUV.u-LeftUV.u)/((r32)XDifference),
                (RightUV.v-LeftUV.v)/((r32)XDifference),
            };
        }
        
        
        __m256 U_8x = _mm256_setr_ps(LeftUV.u + (XOffset + 0.0f)*UVIncrement.u,
                                     LeftUV.u + (XOffset + 1.0f)*UVIncrement.u,
                                     LeftUV.u + (XOffset + 2.0f)*UVIncrement.u,
                                     LeftUV.u + (XOffset + 3.0f)*UVIncrement.u,
                                     LeftUV.u + (XOffset + 4.0f)*UVIncrement.u,
                                     LeftUV.u + (XOffset + 5.0f)*UVIncrement.u,
                                     LeftUV.u + (XOffset + 6.0f)*UVIncrement.u,
                                     LeftUV.u + (XOffset + 7.0f)*UVIncrement.u);
        
        __m256 V_8x = _mm256_setr_ps(LeftUV.v + (XOffset + 0.0f)*UVIncrement.v,
                                     LeftUV.v + (XOffset + 1.0f)*UVIncrement.v,
                                     LeftUV.v + (XOffset + 2.0f)*UVIncrement.v,
                                     LeftUV.v + (XOffset + 3.0f)*UVIncrement.v,
                                     LeftUV.v + (XOffset + 4.0f)*UVIncrement.v,
                                     LeftUV.v + (XOffset + 5.0f)*UVIncrement.v,
                                     LeftUV.v + (XOffset + 6.0f)*UVIncrement.v,
                                     LeftUV.v + (XOffset + 7.0f)*UVIncrement.v);
        
        __m256 UIncrement_8x = _mm256_set1_ps(UVIncrement.u*8.0f);
        __m256 VIncrement_8x = _mm256_set1_ps(UVIncrement.v*8.0f);
        
        v3 NormalIncrement = {};
        if(XDifference != 0)
        {
            NormalIncrement = 
            {
                (RightNormal.x-LeftNormal.x)/((r32)XDifference),
                (RightNormal.y-LeftNormal.y)/((r32)XDifference),
                (RightNormal.z-LeftNormal.z)/((r32)XDifference),
            };
        }
        
        __m256 NormalX_8x = _mm256_setr_ps(LeftNormal.x + (XOffset + 0.0f)*NormalIncrement.x,
                                           LeftNormal.x + (XOffset + 1.0f)*NormalIncrement.x,
                                           LeftNormal.x + (XOffset + 2.0f)*NormalIncrement.x,
                                           LeftNormal.x + (XOffset + 3.0f)*NormalIncrement.x,
                                           LeftNormal.x + (XOffset + 4.0f)*NormalIncrement.x,
                                           LeftNormal.x + (XOffset + 5.0f)*NormalIncrement.x,
                                           LeftNormal.x + (XOffset + 6.0f)*NormalIncrement.x,
                                           LeftNormal.x + (XOffset + 7.0f)*NormalIncrement.x);
        
        __m256 NormalY_8x = _mm256_setr_ps(LeftNormal.y + (XOffset + 0.0f)*NormalIncrement.y,
                                           LeftNormal.y + (XOffset + 1.0f)*NormalIncrement.y,
                                           LeftNormal.y + (XOffset + 2.0f)*NormalIncrement.y,
                                           LeftNormal.y + (XOffset + 3.0f)*NormalIncrement.y,
                                           LeftNormal.y + (XOffset + 4.0f)*NormalIncrement.y,
                                           LeftNormal.y + (XOffset + 5.0f)*NormalIncrement.y,
                                           LeftNormal.y + (XOffset + 6.0f)*NormalIncrement.y,
                                           LeftNormal.y + (XOffset + 7.0f)*NormalIncrement.y);
        
        __m256 NormalZ_8x = _mm256_setr_ps(LeftNormal.z + (XOffset + 0.0f)*NormalIncrement.z,
                                           LeftNormal.z + (XOffset + 1.0f)*NormalIncrement.z,
                                           LeftNormal.z + (XOffset + 2.0f)*NormalIncrement.z,
                                           LeftNormal.z + (XOffset + 3.0f)*NormalIncrement.z,
                                           LeftNormal.z + (XOffset + 4.0f)*NormalIncrement.z,
                                           LeftNormal.z + (XOffset + 5.0f)*NormalIncrement.z,
                                           LeftNormal.z + (XOffset + 6.0f)*NormalIncrement.z,
                                           LeftNormal.z + (XOffset + 7.0f)*NormalIncrement.z);
        
        unproject_simd_result CurrentNormal = NormalizeVector_8x(NormalX_8x, NormalY_8x, NormalZ_8x);
        
        NormalX_8x = CurrentNormal.VertexX;
        NormalY_8x = CurrentNormal.VertexY;
        NormalZ_8x = CurrentNormal.VertexZ;
        
        __m256 NormalXIncrement_8x = _mm256_set1_ps(NormalIncrement.x*8.0f);
        __m256 NormalYIncrement_8x = _mm256_set1_ps(NormalIncrement.y*8.0f);
        __m256 NormalZIncrement_8x = _mm256_set1_ps(NormalIncrement.z*8.0f);
        
        
        v4 ColorIncrement = {};
        if(XDifference != 0)
        {
            ColorIncrement = 
            {
                (RightColor.r-LeftColor.r)/((r32)XDifference),
                (RightColor.g-LeftColor.g)/((r32)XDifference),
                (RightColor.b-LeftColor.b)/((r32)XDifference),
                (RightColor.a-LeftColor.a)/((r32)XDifference),
            };
        }
        
        __m256 ColorR_8x = _mm256_set_ps(LeftColor.r + (XOffset + 0.0f)*ColorIncrement.r,
                                         LeftColor.r + (XOffset + 1.0f)*ColorIncrement.r,
                                         LeftColor.r + (XOffset + 2.0f)*ColorIncrement.r,
                                         LeftColor.r + (XOffset + 3.0f)*ColorIncrement.r,
                                         LeftColor.r + (XOffset + 4.0f)*ColorIncrement.r,
                                         LeftColor.r + (XOffset + 5.0f)*ColorIncrement.r,
                                         LeftColor.r + (XOffset + 6.0f)*ColorIncrement.r,
                                         LeftColor.r + (XOffset + 7.0f)*ColorIncrement.r);
        
        __m256 ColorG_8x = _mm256_set_ps(LeftColor.g + (XOffset + 0.0f)*ColorIncrement.g,
                                         LeftColor.g + (XOffset + 1.0f)*ColorIncrement.g,
                                         LeftColor.g + (XOffset + 2.0f)*ColorIncrement.g,
                                         LeftColor.g + (XOffset + 3.0f)*ColorIncrement.g,
                                         LeftColor.g + (XOffset + 4.0f)*ColorIncrement.g,
                                         LeftColor.g + (XOffset + 5.0f)*ColorIncrement.g,
                                         LeftColor.g + (XOffset + 6.0f)*ColorIncrement.g,
                                         LeftColor.g + (XOffset + 7.0f)*ColorIncrement.g);
        
        __m256 ColorB_8x = _mm256_set_ps(LeftColor.b + (XOffset + 0.0f)*ColorIncrement.b,
                                         LeftColor.b + (XOffset + 1.0f)*ColorIncrement.b,
                                         LeftColor.b + (XOffset + 2.0f)*ColorIncrement.b,
                                         LeftColor.b + (XOffset + 3.0f)*ColorIncrement.b,
                                         LeftColor.b + (XOffset + 4.0f)*ColorIncrement.b,
                                         LeftColor.b + (XOffset + 5.0f)*ColorIncrement.b,
                                         LeftColor.b + (XOffset + 6.0f)*ColorIncrement.b,
                                         LeftColor.b + (XOffset + 7.0f)*ColorIncrement.b);
        
        __m256 ColorA_8x = _mm256_set_ps(LeftColor.a + (XOffset + 0.0f)*ColorIncrement.a,
                                         LeftColor.a + (XOffset + 1.0f)*ColorIncrement.a,
                                         LeftColor.a + (XOffset + 2.0f)*ColorIncrement.a,
                                         LeftColor.a + (XOffset + 3.0f)*ColorIncrement.a,
                                         LeftColor.a + (XOffset + 4.0f)*ColorIncrement.a,
                                         LeftColor.a + (XOffset + 5.0f)*ColorIncrement.a,
                                         LeftColor.a + (XOffset + 6.0f)*ColorIncrement.a,
                                         LeftColor.a + (XOffset + 7.0f)*ColorIncrement.a);
        
        __m256 ColorRIncrement_8x = _mm256_set1_ps(ColorIncrement.r*8.0f);
        __m256 ColorGIncrement_8x = _mm256_set1_ps(ColorIncrement.g*8.0f);
        __m256 ColorBIncrement_8x = _mm256_set1_ps(ColorIncrement.b*8.0f);
        __m256 ColorAIncrement_8x = _mm256_set1_ps(ColorIncrement.a*8.0f);
        
        r32 CurrentZ = CurrentEdgeInList->ZMin;
        r32 ZIncrement = 0.0f;
        
        if(XDifference != 0)
        {
            ZIncrement = (NextEdgeInList->ZMin-CurrentEdgeInList->ZMin)/((r32)XDifference);
        }
        
        __m256 CurrentZ_8x = _mm256_setr_ps(CurrentEdgeInList->ZMin + (XOffset + 0.0f)*ZIncrement,
                                            CurrentEdgeInList->ZMin + (XOffset + 1.0f)*ZIncrement,
                                            CurrentEdgeInList->ZMin + (XOffset + 2.0f)*ZIncrement,
                                            CurrentEdgeInList->ZMin + (XOffset + 3.0f)*ZIncrement,
                                            CurrentEdgeInList->ZMin + (XOffset + 4.0f)*ZIncrement,
                                            CurrentEdgeInList->ZMin + (XOffset + 5.0f)*ZIncrement,
                                            CurrentEdgeInList->ZMin + (XOffset + 6.0f)*ZIncrement,
                                            CurrentEdgeInList->ZMin + (XOffset + 7.0f)*ZIncrement);
        
        __m256 ZIncrement_8x = _mm256_set1_ps(8.0f*ZIncrement);
        
        u8 *Row = (u8 *)Buffer->Memory +
            (s32)LeftX*BITMAP_BYTES_PER_PIXEL +
            RowIndex*Buffer->Pitch;
        
        r32 *ZBufferPixel = (r32 *)ZBuffer +
            (s32)LeftX + RowIndex*ZBufferWidth;
        
        u32 *Pixel = (u32 *)Row;
        __m256i ClipMask_8x = StartClipMask_8x;
        
        u8 *ZMaskPixel = Commands->ZMask + ((s32)LeftX + RowIndex*Commands->Width)/8;
        
        ClipMask_8x = _mm256_set_epi32(ClipMask_8x.m256i_u32[0],
                                       ClipMask_8x.m256i_u32[1],
                                       ClipMask_8x.m256i_u32[2],
                                       ClipMask_8x.m256i_u32[3],
                                       ClipMask_8x.m256i_u32[4],
                                       ClipMask_8x.m256i_u32[5],
                                       ClipMask_8x.m256i_u32[6],
                                       ClipMask_8x.m256i_u32[7]);
        
        for(s32 X = (s32)LeftX;
            X < (s32)RightX;
            X += 8)
        {
            
            void *TextureMemory = Bitmap->Memory;
            
            __m256 FinalU_8x = _mm256_mul_ps(_mm256_div_ps(One_8x, CurrentOneOverZ_8x), U_8x);
            __m256 FinalV_8x = _mm256_mul_ps(_mm256_div_ps(One_8x, CurrentOneOverZ_8x), V_8x);
            
            __m256 TexCoordX_8x = _mm256_set1_ps((r32)Bitmap->Width);
            __m256 TexCoordY_8x = _mm256_set1_ps((r32)Bitmap->Height);
            
            TexCoordX_8x = _mm256_mul_ps(TexCoordX_8x, FinalU_8x);
            TexCoordY_8x = _mm256_mul_ps(TexCoordY_8x, FinalV_8x);
            
            __m256i WriteMask = _mm256_castps_si256(_mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(FinalU_8x, Zero_8x, 13),
                                                                                _mm256_cmp_ps(FinalU_8x, One_8x, 18)),
                                                                  _mm256_and_ps(_mm256_cmp_ps(FinalV_8x, Zero_8x, 13),
                                                                                _mm256_cmp_ps(FinalV_8x, One_8x, 18))));
            
            WriteMask = _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(WriteMask), _mm256_castsi256_ps(ClipMask_8x))); 
            
            //__m256i FetchX_8x = _mm256_cvttps_epi32(TexCoordX_8x); 
            __m128i FetchX_8xA = _mm_cvttps_epi32(_mm_setr_ps(TexCoordX_8x.m256_f32[0],
                                                              TexCoordX_8x.m256_f32[1],
                                                              TexCoordX_8x.m256_f32[2],
                                                              TexCoordX_8x.m256_f32[3]));
            
            __m128i FetchX_8xB = _mm_cvttps_epi32(_mm_setr_ps(TexCoordX_8x.m256_f32[4],
                                                              TexCoordX_8x.m256_f32[5],
                                                              TexCoordX_8x.m256_f32[6],
                                                              TexCoordX_8x.m256_f32[7]));
            
            __m256i FetchY_8x = _mm256_cvttps_epi32(TexCoordY_8x); 
            
            
            //FetchX_8x = _mm256_slli_epi32(FetchX_8x, 2);
            FetchX_8xA = _mm_slli_epi32(FetchX_8xA, 2);
            FetchX_8xB = _mm_slli_epi32(FetchX_8xB, 2);
            
            //FetchY_8x = _mm256_or_si256(_mm256_mullo_epi16(FetchY_8x, TexturePitch_8x),                      
            //                            _mm256_slli_epi32(_mm256_mulhi_epi16(FetchY_8x, TexturePitch_8x), 16));
            
            __m128i FetchY_8xA = _mm_setr_epi32(FetchY_8x.m256i_i32[0],
                                                FetchY_8x.m256i_i32[1],
                                                FetchY_8x.m256i_i32[2],
                                                FetchY_8x.m256i_i32[3]);
            
            __m128i FetchY_8xB = _mm_setr_epi32(FetchY_8x.m256i_i32[4],
                                                FetchY_8x.m256i_i32[5],
                                                FetchY_8x.m256i_i32[6],
                                                FetchY_8x.m256i_i32[7]);
            
            __m128i TexturePitch_8xA = _mm_set1_epi32(TexturePitch_8x.m256i_u32[0]);
            
            __m128i TexturePitch_8xB = _mm_set1_epi32(TexturePitch_8x.m256i_u32[0]);
            
            FetchY_8xA = _mm_or_si128(_mm_mullo_epi16(FetchY_8xA, TexturePitch_8xA),                      
                                      _mm_slli_epi32(_mm_mulhi_epi16(FetchY_8xA, TexturePitch_8xA), 16));
            
            FetchY_8xB = _mm_or_si128(_mm_mullo_epi16(FetchY_8xB, TexturePitch_8xB),                      
                                      _mm_slli_epi32(_mm_mulhi_epi16(FetchY_8xB, TexturePitch_8xB), 16));
            
            //__m256i Fetch_8x = _mm256_add_epi32(FetchX_8x, FetchY_8x);
            __m128i Fetch_8xA = _mm_add_epi32(FetchX_8xA, FetchY_8xA);
            __m128i Fetch_8xB = _mm_add_epi32(FetchX_8xB, FetchY_8xB);
            
            __m256i Fetch_8x = _mm256_set_epi32(Fetch_8xA.m128i_i32[0],
                                                Fetch_8xA.m128i_i32[1],
                                                Fetch_8xA.m128i_i32[2],
                                                Fetch_8xA.m128i_i32[3],
                                                Fetch_8xB.m128i_i32[0],
                                                Fetch_8xB.m128i_i32[1],
                                                Fetch_8xB.m128i_i32[2],
                                                Fetch_8xB.m128i_i32[3]);
            
            s32 Fetch0 = Fetch_8x.m256i_u32[0];
            s32 Fetch1 = Fetch_8x.m256i_u32[1];
            s32 Fetch2 = Fetch_8x.m256i_u32[2];
            s32 Fetch3 = Fetch_8x.m256i_u32[3];
            s32 Fetch4 = Fetch_8x.m256i_u32[4];
            s32 Fetch5 = Fetch_8x.m256i_u32[5];
            s32 Fetch6 = Fetch_8x.m256i_u32[6];
            s32 Fetch7 = Fetch_8x.m256i_u32[7];
            
            u8 *TexelPtr0 = ((u8 *)TextureMemory) + Fetch0;
            u8 *TexelPtr1 = ((u8 *)TextureMemory) + Fetch1;
            u8 *TexelPtr2 = ((u8 *)TextureMemory) + Fetch2;
            u8 *TexelPtr3 = ((u8 *)TextureMemory) + Fetch3;
            u8 *TexelPtr4 = ((u8 *)TextureMemory) + Fetch4;
            u8 *TexelPtr5 = ((u8 *)TextureMemory) + Fetch5;
            u8 *TexelPtr6 = ((u8 *)TextureMemory) + Fetch6;
            u8 *TexelPtr7 = ((u8 *)TextureMemory) + Fetch7;
            
#if 0
            __m256i Sample = _mm256_setr_epi32(*(u32 *)(TexelPtr0),
                                               *(u32 *)(TexelPtr1),
                                               *(u32 *)(TexelPtr2),
                                               *(u32 *)(TexelPtr3),
                                               *(u32 *)(TexelPtr4),
                                               *(u32 *)(TexelPtr5),
                                               *(u32 *)(TexelPtr6),
                                               *(u32 *)(TexelPtr7));
#endif
            
            __m128i SampleA = _mm_setr_epi32(*(u32 *)(TexelPtr0),
                                             *(u32 *)(TexelPtr1),
                                             *(u32 *)(TexelPtr2),
                                             *(u32 *)(TexelPtr3));
            
            __m128i SampleB = _mm_setr_epi32(*(u32 *)(TexelPtr4),
                                             *(u32 *)(TexelPtr5),
                                             *(u32 *)(TexelPtr6),
                                             *(u32 *)(TexelPtr7));
            
            __m128i MaskFF_4x = _mm_set1_epi32(0xFF);
            
            __m128i SampleAR = _mm_and_si128(_mm_srli_epi32(SampleA, 24), MaskFF_4x);
            __m128i SampleBR = _mm_and_si128(_mm_srli_epi32(SampleB, 24), MaskFF_4x);
            
            __m128i SampleAG = _mm_and_si128(_mm_srli_epi32(SampleA, 16), MaskFF_4x);
            __m128i SampleBG = _mm_and_si128(_mm_srli_epi32(SampleB, 16), MaskFF_4x);
            
            __m128i SampleAB = _mm_and_si128(_mm_srli_epi32(SampleA, 8), MaskFF_4x);
            __m128i SampleBB = _mm_and_si128(_mm_srli_epi32(SampleB, 8), MaskFF_4x);
            
            __m128i SampleAA = _mm_and_si128(_mm_srli_epi32(SampleA, 0), MaskFF_4x);
            __m128i SampleBA = _mm_and_si128(_mm_srli_epi32(SampleB, 0), MaskFF_4x);
            
            __m256i SampleFinalR = _mm256_set_epi32(SampleAR.m128i_u32[0],
                                                    SampleAR.m128i_u32[1],
                                                    SampleAR.m128i_u32[2],
                                                    SampleAR.m128i_u32[3],
                                                    SampleBR.m128i_u32[0],
                                                    SampleBR.m128i_u32[1],
                                                    SampleBR.m128i_u32[2],
                                                    SampleBR.m128i_u32[3]);
            
            __m256i SampleFinalG = _mm256_set_epi32(SampleAG.m128i_u32[0],
                                                    SampleAG.m128i_u32[1],
                                                    SampleAG.m128i_u32[2],
                                                    SampleAG.m128i_u32[3],
                                                    SampleBG.m128i_u32[0],
                                                    SampleBG.m128i_u32[1],
                                                    SampleBG.m128i_u32[2],
                                                    SampleBG.m128i_u32[3]);
            
            __m256i SampleFinalB = _mm256_set_epi32(SampleAB.m128i_u32[0],
                                                    SampleAB.m128i_u32[1],
                                                    SampleAB.m128i_u32[2],
                                                    SampleAB.m128i_u32[3],
                                                    SampleBB.m128i_u32[0],
                                                    SampleBB.m128i_u32[1],
                                                    SampleBB.m128i_u32[2],
                                                    SampleBB.m128i_u32[3]);
            
            __m256i SampleFinalA = _mm256_set_epi32(SampleAA.m128i_u32[0],
                                                    SampleAA.m128i_u32[1],
                                                    SampleAA.m128i_u32[2],
                                                    SampleAA.m128i_u32[3],
                                                    SampleBA.m128i_u32[0],
                                                    SampleBA.m128i_u32[1],
                                                    SampleBA.m128i_u32[2],
                                                    SampleBA.m128i_u32[3]);
            
            //ColorA_8x = _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(Sample, 24), MaskFF_8x)), One255_8x);
            //ColorR_8x = _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(Sample, 16), MaskFF_8x)), One255_8x);
            //ColorG_8x = _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(Sample, 8), MaskFF_8x)), One255_8x);
            //ColorB_8x = _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(Sample, 0), MaskFF_8x)), One255_8x);
            
            ColorA_8x = _mm256_div_ps(_mm256_cvtepi32_ps(SampleFinalR), One255_8x);
            ColorR_8x = _mm256_div_ps(_mm256_cvtepi32_ps(SampleFinalG), One255_8x);
            ColorG_8x = _mm256_div_ps(_mm256_cvtepi32_ps(SampleFinalB), One255_8x);
            ColorB_8x = _mm256_div_ps(_mm256_cvtepi32_ps(SampleFinalA), One255_8x);
            
            
            __m256 FinalColorR_8x = _mm256_set1_ps(0.0f);
            __m256 FinalColorG_8x = _mm256_set1_ps(0.0f);
            __m256 FinalColorB_8x = _mm256_set1_ps(0.0f);
            __m256 FinalColorA_8x = _mm256_set1_ps(0.0f);
            
            if(PhongShading)
            {
                light_data *Lights = &Commands->LightData; 
                light_info *Light = Lights->Lights;
                
                unproject_simd_result CurrentVertexP = 
                    UnprojectVertex_8x((r32)X, (r32)RowIndex, CurrentZ_8x, &Commands->Transform);
                
                for(u32 LightIndex = 0;
                    LightIndex < Lights->LightCount;
                    ++LightIndex, ++Light)
                {
                    if(LightIndex == 0)
                    {
                        FinalColorR_8x = _mm256_mul_ps(ColorR_8x, _mm256_set1_ps(Lights->AmbientIntensity.r));
                        FinalColorG_8x = _mm256_mul_ps(ColorG_8x, _mm256_set1_ps(Lights->AmbientIntensity.g));
                        FinalColorB_8x = _mm256_mul_ps(ColorB_8x, _mm256_set1_ps(Lights->AmbientIntensity.b));
                        FinalColorA_8x = _mm256_mul_ps(ColorA_8x, _mm256_set1_ps(Lights->AmbientIntensity.a));
                    }
                    
                    __m256 LightX = _mm256_set1_ps(Light->P.x);
                    __m256 LightY = _mm256_set1_ps(Light->P.y);
                    __m256 LightZ = _mm256_set1_ps(Light->P.z);
                    
                    __m256 VectorToLightX_8x = _mm256_sub_ps(LightX, CurrentVertexP.VertexX);
                    __m256 VectorToLightY_8x = _mm256_sub_ps(LightY, CurrentVertexP.VertexY);
                    __m256 VectorToLightZ_8x = _mm256_sub_ps(LightZ, CurrentVertexP.VertexZ);
                    
                    unproject_simd_result VectorToLight = NormalizeVector_8x(VectorToLightX_8x, VectorToLightY_8x, VectorToLightZ_8x);
                    
                    unproject_simd_result LightDirection = VectorToLight;
                    
                    __m256 DotNormalLightX_8x = _mm256_mul_ps(NormalX_8x, VectorToLight.VertexX);
                    __m256 DotNormalLightY_8x = _mm256_mul_ps(NormalY_8x, VectorToLight.VertexY);
                    __m256 DotNormalLightZ_8x = _mm256_mul_ps(NormalZ_8x, VectorToLight.VertexZ);
                    
                    __m256 CosineOfIncidence_8x = _mm256_min_ps(One_8x, _mm256_max_ps(Zero_8x, _mm256_add_ps(_mm256_add_ps(DotNormalLightX_8x, DotNormalLightY_8x), DotNormalLightZ_8x)));
                    
                    unproject_simd_result ViewDirection = NormalizeVector_8x(
                        _mm256_sub_ps(Zero_8x, CurrentVertexP.VertexX), 
                        _mm256_sub_ps(Zero_8x, CurrentVertexP.VertexY), 
                        _mm256_sub_ps(Zero_8x, CurrentVertexP.VertexZ));
                    
                    __m256 HalfAngleX_8x = _mm256_add_ps(LightDirection.VertexX, ViewDirection.VertexX);
                    __m256 HalfAngleY_8x = _mm256_add_ps(LightDirection.VertexY, ViewDirection.VertexY);
                    __m256 HalfAngleZ_8x = _mm256_add_ps(LightDirection.VertexZ, ViewDirection.VertexZ);
                    
                    unproject_simd_result HalfAngle = NormalizeVector_8x(HalfAngleX_8x, HalfAngleY_8x, HalfAngleZ_8x);
                    
                    __m256 PhongTerm_8x = _mm256_min_ps(One_8x, _mm256_max_ps(Zero_8x, _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(NormalX_8x, HalfAngle.VertexX), _mm256_mul_ps(NormalY_8x, HalfAngle.VertexY)), _mm256_mul_ps(NormalZ_8x, HalfAngle.VertexZ))));
                    
                    for(u32 FactorIndex = 0;
                        FactorIndex < 4;
                        ++FactorIndex)
                    {
                        PhongTerm_8x = _mm256_mul_ps(PhongTerm_8x, PhongTerm_8x);
                    }
                    
                    __m256 SpecularColorR_8x = _mm256_set1_ps(1.0f); 
                    __m256 SpecularColorG_8x = _mm256_set1_ps(1.0f); 
                    __m256 SpecularColorB_8x = _mm256_set1_ps(1.0f); 
                    __m256 SpecularColorA_8x = _mm256_set1_ps(1.0f); 
                    
                    __m256 LightIntensityR_8x = _mm256_set1_ps(Light->Intensity.r); 
                    __m256 LightIntensityG_8x = _mm256_set1_ps(Light->Intensity.g); 
                    __m256 LightIntensityB_8x = _mm256_set1_ps(Light->Intensity.b); 
                    __m256 LightIntensityA_8x = _mm256_set1_ps(Light->Intensity.a); 
                    
                    FinalColorR_8x = _mm256_add_ps(FinalColorR_8x, 
                                                   _mm256_add_ps(_mm256_mul_ps(CosineOfIncidence_8x, _mm256_mul_ps(ColorR_8x, LightIntensityR_8x)), 
                                                                 _mm256_mul_ps(PhongTerm_8x, _mm256_mul_ps(SpecularColorR_8x, LightIntensityR_8x))));
                    
                    FinalColorG_8x = _mm256_add_ps(FinalColorG_8x, 
                                                   _mm256_add_ps(_mm256_mul_ps(CosineOfIncidence_8x, _mm256_mul_ps(ColorG_8x, LightIntensityG_8x)), 
                                                                 _mm256_mul_ps(PhongTerm_8x, _mm256_mul_ps(SpecularColorG_8x, LightIntensityG_8x))));
                    
                    FinalColorB_8x = _mm256_add_ps(FinalColorB_8x, 
                                                   _mm256_add_ps(_mm256_mul_ps(CosineOfIncidence_8x, _mm256_mul_ps(ColorB_8x, LightIntensityB_8x)), 
                                                                 _mm256_mul_ps(PhongTerm_8x, _mm256_mul_ps(SpecularColorB_8x, LightIntensityB_8x))));
                    
                    FinalColorA_8x = _mm256_add_ps(FinalColorA_8x, 
                                                   _mm256_add_ps(_mm256_mul_ps(CosineOfIncidence_8x, _mm256_mul_ps(ColorA_8x, LightIntensityA_8x)),
                                                                 _mm256_mul_ps(PhongTerm_8x, _mm256_mul_ps(SpecularColorA_8x, LightIntensityA_8x))));
                }
                
                FinalColorR_8x = _mm256_max_ps(_mm256_min_ps(FinalColorR_8x, One_8x), Zero_8x);
                FinalColorG_8x = _mm256_max_ps(_mm256_min_ps(FinalColorG_8x, One_8x), Zero_8x);
                FinalColorB_8x = _mm256_max_ps(_mm256_min_ps(FinalColorB_8x, One_8x), Zero_8x);
                FinalColorA_8x = _mm256_max_ps(_mm256_min_ps(FinalColorA_8x, One_8x), Zero_8x);
                
                //__m256i Color32_8x = _mm256_or_si256(_mm256_or_si256(_mm256_or_si256(_mm256_srli_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(FinalColorG_8x, One255_8x)), 8), 
                //                                                                     _mm256_cvtps_epi32(_mm256_mul_ps(FinalColorG_8x, One255_8x))),
                //                                                     _mm256_srli_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(FinalColorR_8x, One255_8x)), 16)),
                //                                     _mm256_srli_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(FinalColorA_8x, One255_8x)), 24));
                
                __m256i Color32R_8x = _mm256_cvtps_epi32(_mm256_mul_ps(FinalColorR_8x, One255_8x));
                __m256i Color32G_8x = _mm256_cvtps_epi32(_mm256_mul_ps(FinalColorG_8x, One255_8x));
                __m256i Color32B_8x = _mm256_cvtps_epi32(_mm256_mul_ps(FinalColorB_8x, One255_8x));
                __m256i Color32A_8x = _mm256_cvtps_epi32(_mm256_mul_ps(FinalColorA_8x, One255_8x));
                
                __m128i Color32R_8xA = _mm_set_epi32(Color32R_8x.m256i_u32[0],
                                                     Color32R_8x.m256i_u32[1],
                                                     Color32R_8x.m256i_u32[2],
                                                     Color32R_8x.m256i_u32[3]);
                
                __m128i Color32R_8xB = _mm_set_epi32(Color32R_8x.m256i_u32[4],
                                                     Color32R_8x.m256i_u32[5],
                                                     Color32R_8x.m256i_u32[6],
                                                     Color32R_8x.m256i_u32[7]);
                
                __m128i Color32G_8xA = _mm_set_epi32(Color32G_8x.m256i_u32[0],
                                                     Color32G_8x.m256i_u32[1],
                                                     Color32G_8x.m256i_u32[2],
                                                     Color32G_8x.m256i_u32[3]);
                
                __m128i Color32G_8xB = _mm_set_epi32(Color32G_8x.m256i_u32[4],
                                                     Color32G_8x.m256i_u32[5],
                                                     Color32G_8x.m256i_u32[6],
                                                     Color32G_8x.m256i_u32[7]);
                
                __m128i Color32B_8xA = _mm_set_epi32(Color32B_8x.m256i_u32[0],
                                                     Color32B_8x.m256i_u32[1],
                                                     Color32B_8x.m256i_u32[2],
                                                     Color32B_8x.m256i_u32[3]);
                
                __m128i Color32B_8xB = _mm_set_epi32(Color32B_8x.m256i_u32[4],
                                                     Color32B_8x.m256i_u32[5],
                                                     Color32B_8x.m256i_u32[6],
                                                     Color32B_8x.m256i_u32[7]);
                
                __m128i Color32A_8xA = _mm_set_epi32(Color32A_8x.m256i_u32[0],
                                                     Color32A_8x.m256i_u32[1],
                                                     Color32A_8x.m256i_u32[2],
                                                     Color32A_8x.m256i_u32[3]);
                
                __m128i Color32A_8xB = _mm_set_epi32(Color32A_8x.m256i_u32[4],
                                                     Color32A_8x.m256i_u32[5],
                                                     Color32A_8x.m256i_u32[6],
                                                     Color32A_8x.m256i_u32[7]);
                
                Color32R_8xA = _mm_slli_epi32(Color32R_8xA, 16);
                Color32R_8xB = _mm_slli_epi32(Color32R_8xB, 16);
                Color32G_8xA = _mm_slli_epi32(Color32G_8xA, 8);
                Color32G_8xB = _mm_slli_epi32(Color32G_8xB, 8);
                Color32B_8xA = _mm_slli_epi32(Color32B_8xA, 0);
                Color32B_8xB = _mm_slli_epi32(Color32B_8xB, 0);
                Color32A_8xA = _mm_slli_epi32(Color32A_8xA, 24);
                Color32A_8xB = _mm_slli_epi32(Color32A_8xB, 24);
                
                __m128i Color32_8xA = _mm_or_si128(_mm_or_si128(_mm_or_si128(Color32R_8xA, Color32G_8xA), Color32B_8xA), Color32A_8xA);
                __m128i Color32_8xB = _mm_or_si128(_mm_or_si128(_mm_or_si128(Color32R_8xB, Color32G_8xB), Color32B_8xB), Color32A_8xB);
                
                __m256i Color32_8x = _mm256_set_epi32(Color32_8xB.m128i_u32[0],
                                                      Color32_8xB.m128i_u32[1],
                                                      Color32_8xB.m128i_u32[2],
                                                      Color32_8xB.m128i_u32[3],
                                                      Color32_8xA.m128i_u32[0],
                                                      Color32_8xA.m128i_u32[1],
                                                      Color32_8xA.m128i_u32[2],
                                                      Color32_8xA.m128i_u32[3]
                                                      );
                
                u8 volatile FinalMask = 1;
                
                __m256i FinalWriteMask = WriteMask;
                __m256i FinalColor32_8x = Color32_8x;
                __m256i DebugColor32_8x = _mm256_set1_epi8(-1);
                
                while(FinalMask != 0)
                {
                    
                    FinalMask = _InterlockedCompareExchange8((char volatile *)ZMaskPixel,
                                                             1,
                                                             0);
                    
                    if(FinalMask == 0)
                    {
                        
                        __m256 OriginalZ_8x = _mm256_load_ps(ZBufferPixel); 
                        __m256 ZMask = _mm256_cmp_ps(CurrentZ_8x, OriginalZ_8x, 30);
                        
                        ZMask = _mm256_and_ps(ZMask, _mm256_castsi256_ps(WriteMask));
                        
                        FinalWriteMask = _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(WriteMask), ZMask));
                        
                        __m256 NewZ_8x = _mm256_or_ps(_mm256_and_ps(ZMask, CurrentZ_8x),
                                                      _mm256_andnot_ps(ZMask, OriginalZ_8x));
                        
                        __m256i OriginalDest = _mm256_loadu_si256((__m256i *)Pixel);
                        
                        FinalColor32_8x = _mm256_castps_si256(_mm256_or_ps(_mm256_and_ps(_mm256_castsi256_ps(FinalWriteMask), _mm256_castsi256_ps(FinalColor32_8x)),
                                                                           _mm256_andnot_ps(_mm256_castsi256_ps(FinalWriteMask), _mm256_castsi256_ps(OriginalDest))));
                        
                        _mm256_storeu_ps(ZBufferPixel, NewZ_8x);
                        _mm256_storeu_si256((__m256i *)Pixel, FinalColor32_8x);
                        _WriteBarrier();
                        
                        *ZMaskPixel = 0;
                    }
                }
                
                if((X + 16) < RightX)
                {                  
                    ClipMask_8x = _mm256_set1_epi8(-1);
                }
                else
                {                  
                    ClipMask_8x = EndClipMask_8x;
                    ClipMask_8x = _mm256_set_epi32(ClipMask_8x.m256i_u32[0],
                                                   ClipMask_8x.m256i_u32[1],
                                                   ClipMask_8x.m256i_u32[2],
                                                   ClipMask_8x.m256i_u32[3],
                                                   ClipMask_8x.m256i_u32[4],
                                                   ClipMask_8x.m256i_u32[5],
                                                   ClipMask_8x.m256i_u32[6],
                                                   ClipMask_8x.m256i_u32[7]);
                }
                
                ZBufferPixel += 8;
                Pixel += 8;
                ZMaskPixel++;
                
                unproject_simd_result NewNormals = NormalizeVector_8x(
                    _mm256_add_ps(NormalX_8x, NormalXIncrement_8x),
                    _mm256_add_ps(NormalY_8x, NormalYIncrement_8x),
                    _mm256_add_ps(NormalZ_8x, NormalZIncrement_8x)
                    );
                
                NormalX_8x = NewNormals.VertexX;
                NormalY_8x = NewNormals.VertexY;
                NormalZ_8x = NewNormals.VertexZ;
                
                ColorR_8x = _mm256_add_ps(ColorR_8x, ColorRIncrement_8x);
                ColorG_8x = _mm256_add_ps(ColorG_8x, ColorGIncrement_8x);
                ColorB_8x = _mm256_add_ps(ColorB_8x, ColorBIncrement_8x);
                ColorA_8x = _mm256_add_ps(ColorA_8x, ColorAIncrement_8x);
                
                CurrentZ_8x = _mm256_add_ps(CurrentZ_8x, ZIncrement_8x);
                
                CurrentOneOverZ_8x = _mm256_add_ps(CurrentOneOverZ_8x, OneOverZIncrement_8x);
                
                U_8x = _mm256_add_ps(U_8x, UIncrement_8x);
                V_8x = _mm256_add_ps(V_8x, VIncrement_8x);
                
            }
            else
            {
                
                
                FinalColorR_8x = ColorR_8x;
                FinalColorR_8x = ColorG_8x;
                FinalColorR_8x = ColorB_8x;
                FinalColorR_8x = ColorA_8x;
                
                __m256i Color32_8x = _mm256_or_si256(_mm256_or_si256(_mm256_or_si256(_mm256_srli_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(FinalColorG_8x, One255_8x)), 8), 
                                                                                     _mm256_cvtps_epi32(_mm256_mul_ps(FinalColorG_8x, One255_8x))),
                                                                     _mm256_srli_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(FinalColorR_8x, One255_8x)), 16)),
                                                     _mm256_srli_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(FinalColorA_8x, One255_8x)), 24));
                
                if(CurrentZ > *ZBufferPixel)
                {
                    _mm256_storeu_ps(ZBufferPixel, CurrentZ_8x);
                    _mm256_storeu_si256((__m256i *)Pixel, Color32_8x);
                }
                ZBufferPixel += 8;
                Pixel += 8;
                
                
                
                ColorR_8x = _mm256_add_ps(ColorR_8x, ColorRIncrement_8x);
                ColorG_8x = _mm256_add_ps(ColorG_8x, ColorGIncrement_8x);
                ColorB_8x = _mm256_add_ps(ColorB_8x, ColorBIncrement_8x);
                ColorA_8x = _mm256_add_ps(ColorA_8x, ColorAIncrement_8x);
                
                CurrentZ_8x = _mm256_add_ps(CurrentZ_8x, ZIncrement_8x);
                
            }
            
        }
    }    
}

internal void *
AddWorkToThreadMemory(game_render_commands *Commands, void *Work, u32 Size)
{
    void *Result = (u8 *)Commands->ThreadMemory + Commands->ThreadMemorySizeUsed;
    
    Assert(Commands->ThreadMemorySizeUsed + Size < Commands->ThreadMemorySize);
    
    Copy(Size, Work, Result);
    
    Commands->ThreadMemorySizeUsed += Size;
    
    return Result;
}

internal PLATFORM_WORK_QUEUE_CALLBACK(DoLineRenderWork)
{
    line_render_work *Work = (line_render_work *)Data;
    
    FillLineOptimized(Work->Commands, Work->OutputTarget, Work->Bitmap, &Work->CurrentEdgeInList, &Work->NextEdgeInList, Work->RowIndex, Work->PhongShading);
}

internal PLATFORM_WORK_QUEUE_CALLBACK(DoBufferLineRenderWork)
{
    buffer_line_render_work *Work = (buffer_line_render_work *)Data;
    
    FillLinesOptimized(Work->Commands, Work->OutputTarget, Work->Bitmap, &Work->Edges, Work->EdgeCount, Work->RowIndex, Work->PhongShading);
}

internal void
DrawModelOptimized(loaded_bitmap *Buffer,
             edge_info *Edges, u32 EdgeCount,
             game_render_commands *Commands,
             loaded_bitmap *Bitmap = 0,
             b32 PhongShading = 0)
{
    r32 *ZBuffer = Commands->ZBuffer;
    u32 ZBufferWidth = Commands->Width;
    
    s32 FirstRow = Edges[0].YMin;
    
    s32 Height;
    s32 MaxRow = Edges[0].YMax;
    for(u32 EdgeIndex = 1;
        EdgeIndex < EdgeCount;
        ++EdgeIndex)
    {
        if(MaxRow < Edges[EdgeIndex].YMax)
        {
            MaxRow = Edges[EdgeIndex].YMax;
        }
    }
    
    Height = MaxRow - FirstRow;

    edge_info *ListHead = 0;
    edge_info *ListTail = 0;
    
    s32 MaxY = FirstRow + Height;
    if(MaxY > Buffer->Height)
    {
        MaxY = Buffer->Height;
    }
    
    for(s32 RowIndex = FirstRow;
        RowIndex < MaxY;
        ++RowIndex)
    {
        for(u32 EdgeIndex = 0;
            EdgeIndex < EdgeCount;
            ++EdgeIndex)
        {
            edge_info *CurrentEdge = Edges + EdgeIndex;
            
            if(CurrentEdge->YMin == RowIndex)
            {
                if(ListHead)
                {
                    if(CurrentEdge->XMin < ListHead->XMin || 
                       (CurrentEdge->XMin == ListHead->XMin &&
                        (CurrentEdge->Gradient < ListHead->Gradient ||
                         (CurrentEdge->Gradient == ListHead->Gradient &&
                          CurrentEdge->Left < ListHead->Left))))
                    {
                        CurrentEdge->Next = ListHead;
                        ListHead = CurrentEdge;
                    }
                    else
                    {
                        edge_info *ComparedEdge = ListHead;
                        edge_info *PreviousEdge = ListHead;
                        while(ComparedEdge != ListTail)
                        {
                            ComparedEdge = ComparedEdge->Next;
                            
                            if(CurrentEdge->XMin < ComparedEdge->XMin || 
                               (CurrentEdge->XMin == ComparedEdge->XMin &&
                                (CurrentEdge->Gradient < ComparedEdge->Gradient ||
                                 (CurrentEdge->Gradient == ComparedEdge->Gradient &&
                                  CurrentEdge->Left < ComparedEdge->Left))))
                            {
                                CurrentEdge->Next = ComparedEdge;
                                PreviousEdge->Next = CurrentEdge;
                                ComparedEdge = ListTail;
                            }
                            else
                            {
                                PreviousEdge = ComparedEdge;
                            }
                            
                        }                        
                        
                        if(PreviousEdge == ComparedEdge)
                        {
                            ListTail->Next = CurrentEdge;
                            ListTail = CurrentEdge;
                        }
                    }

                }
                else
                {
                    ListHead = CurrentEdge;
                    ListTail = ListHead;
                }
            }
        }
        
        while(ListHead->YMax <= RowIndex)
        {
            edge_info *RemovedEdge = ListHead;
            ListHead = ListHead->Next;
            RemovedEdge->Next = 0;
        }
        
        edge_info *PreviousEdge = ListHead;
        edge_info *CheckedEdge = ListHead;
        
        if(CheckedEdge)
        {
            while(CheckedEdge != ListTail)
            {
                CheckedEdge = CheckedEdge->Next;
                
                if(CheckedEdge->YMax <= RowIndex)
                {
                    
                    if(CheckedEdge == ListTail)
                    {
                        ListTail = PreviousEdge;
                        ListTail->Next = 0;
                        CheckedEdge = ListTail;
                    }
                    else
                    {
                        PreviousEdge->Next = CheckedEdge->Next;
                        CheckedEdge = PreviousEdge;
                    }
                }
                
                PreviousEdge = CheckedEdge;
            }
        }
        
        edge_info *PreviousCurrentEdgeInList = 0;
        edge_info *PreviousNextEdgeInList = 0;
        edge_info *CurrentEdgeInList = ListHead;
        edge_info *NextEdgeInList = CurrentEdgeInList->Next;
        
        __m256 One_8x = _mm256_set1_ps(1.0f);
        __m256 One255_8x = _mm256_set1_ps(255.0f);
        __m256 Half_8x = _mm256_set1_ps(0.5f);
        __m256 Zero_8x = _mm256_set1_ps(0.0f);
        __m256i TexturePitch_8x = _mm256_set1_epi32(Bitmap->Pitch);
        __m256i MaskFF_8x = _mm256_set1_epi32(0xFF);
        __m256i MaskFFFF_8x = _mm256_set1_epi32(0xFFFF);      
        __m256i MaskFF00FF_8x = _mm256_set1_epi32(0x00FF00FF);
        
        while(NextEdgeInList != 0)
        {
            r32 XOffset = 0.0f;
            if(RowIndex >= 0)
            {
                
                r32 LeftX = CurrentEdgeInList->XMin;
                if(LeftX < 0)
                {
                    XOffset = (r32)-XOffset;
                    LeftX = 0;
                }
                else if(LeftX >= Buffer->Width)
                {
                    LeftX = (r32)Buffer->Width - 1;
                }
                
                r32 RightX = NextEdgeInList->XMin;
                if(RightX < 0)
                {
                    RightX = 0;
                }
                else if(RightX >= Buffer->Width)
                {
                    RightX = (r32)Buffer->Width - 1;
                }
                
                s32 ColorLeftPosition = RoundR32ToS32(CurrentEdgeInList->XMin);
                s32 ColorRightPosition = RoundR32ToS32(NextEdgeInList->XMin);
                s32 XDifference = ColorRightPosition - ColorLeftPosition;
                
                v4 LeftColor = CurrentEdgeInList->MinColor; 
                v4 RightColor = NextEdgeInList->MinColor; 

                v3 LeftNormal = CurrentEdgeInList->MinNormal;
                v3 RightNormal = NextEdgeInList->MinNormal;
                
                v2 LeftUV = V2(CurrentEdgeInList->UMin,
                               CurrentEdgeInList->VMin);
                
                v2 RightUV = V2(NextEdgeInList->UMin,
                               NextEdgeInList->VMin);
                
                r32 LeftOneOverZ = CurrentEdgeInList->OneOverZMin;
                r32 RightOneOverZ = NextEdgeInList->OneOverZMin;
                

                __m128i StartClipMask_4x = _mm_set1_epi8(-1);
                __m128i EndClipMask_4x = _mm_set1_epi8(-1);
                
                clip_mask_pair StartClipMaskPairs[] =
                {
                    
                    {_mm_slli_si128(StartClipMask_4x, 0), _mm_slli_si128(StartClipMask_4x, 0)},
                    {_mm_slli_si128(StartClipMask_4x, 0), _mm_slli_si128(StartClipMask_4x, 4)},
                    {_mm_slli_si128(StartClipMask_4x, 0), _mm_slli_si128(StartClipMask_4x, 8)},
                    {_mm_slli_si128(StartClipMask_4x, 0), _mm_slli_si128(StartClipMask_4x, 12)},
                    {_mm_slli_si128(StartClipMask_4x, 0), _mm_slli_si128(StartClipMask_4x, 16)},
                    {_mm_slli_si128(StartClipMask_4x, 4), _mm_slli_si128(StartClipMask_4x, 16)},
                    {_mm_slli_si128(StartClipMask_4x, 8), _mm_slli_si128(StartClipMask_4x, 16)},
                    {_mm_slli_si128(StartClipMask_4x, 12), _mm_slli_si128(StartClipMask_4x, 16)},
                    
                };
                
                clip_mask_pair EndClipMaskPairs[] =
                {
                    {_mm_srli_si128(EndClipMask_4x, 16), _mm_srli_si128(EndClipMask_4x, 16)},
                    {_mm_srli_si128(EndClipMask_4x, 16), _mm_srli_si128(EndClipMask_4x, 12)},
                    {_mm_srli_si128(EndClipMask_4x, 16), _mm_srli_si128(EndClipMask_4x, 8)},
                    {_mm_srli_si128(EndClipMask_4x, 16), _mm_srli_si128(EndClipMask_4x, 4)},
                    {_mm_srli_si128(EndClipMask_4x, 16), _mm_srli_si128(EndClipMask_4x, 0)},
                    {_mm_srli_si128(EndClipMask_4x, 12), _mm_srli_si128(EndClipMask_4x, 0)},
                    {_mm_srli_si128(EndClipMask_4x, 8), _mm_srli_si128(EndClipMask_4x, 0)},
                    {_mm_srli_si128(EndClipMask_4x, 4), _mm_srli_si128(EndClipMask_4x, 0)},
                };
                
                LeftX = (r32)RoundR32ToS32(LeftX);
                RightX = (r32)RoundR32ToS32(RightX);
                
                s32 MinX = (s32)LeftX;
                s32 MaxX = (s32)RightX;
                
                __m256i StartClipMask_8x = _mm256_set1_epi32(-1);
                __m256i EndClipMask_8x = _mm256_set1_epi32(-1);
                
                if(MinX & 7)
                {
                    clip_mask_pair Pair = StartClipMaskPairs[MinX & 7];
                    StartClipMask_8x = _mm256_set_epi32(Pair.SecondHalf.m128i_u32[0],
                                                      Pair.SecondHalf.m128i_u32[1],
                                                      Pair.SecondHalf.m128i_u32[2],
                                                      Pair.SecondHalf.m128i_u32[3],
                                                      Pair.FirstHalf.m128i_u32[0],
                                                      Pair.FirstHalf.m128i_u32[1],
                                                      Pair.FirstHalf.m128i_u32[2],
                                                      Pair.FirstHalf.m128i_u32[3]);
                    LeftX = (r32)(MinX & ~7);
                    XOffset -= (r32)(MinX & 7)*1.0f;
                }
                
                if(MaxX & 7)
                {
                    clip_mask_pair Pair = EndClipMaskPairs[MaxX & 7];
                    EndClipMask_8x = _mm256_set_epi32(Pair.SecondHalf.m128i_u32[0],
                                                    Pair.SecondHalf.m128i_u32[1],
                                                    Pair.SecondHalf.m128i_u32[2],
                                                    Pair.SecondHalf.m128i_u32[3],
                                                    Pair.FirstHalf.m128i_u32[0],
                                                    Pair.FirstHalf.m128i_u32[1],
                                                    Pair.FirstHalf.m128i_u32[2],
                                                    Pair.FirstHalf.m128i_u32[3]);
                    RightX = (r32)((MaxX & ~7) + 8);
                }
                
                
                #if 1
                if((LeftX + 8) >= RightX)
                {
                    
                    __m128i StartClipMask_8xA = _mm_set_epi32(StartClipMask_8x.m256i_u32[0],
                                                               StartClipMask_8x.m256i_u32[1],
                                                               StartClipMask_8x.m256i_u32[2],
                                                               StartClipMask_8x.m256i_u32[3]);
                    
                    __m128i StartClipMask_8xB = _mm_set_epi32(StartClipMask_8x.m256i_u32[4],
                                                               StartClipMask_8x.m256i_u32[5],
                                                               StartClipMask_8x.m256i_u32[6],
                                                               StartClipMask_8x.m256i_u32[7]);
                    
                    __m128i EndClipMask_8xA = _mm_set_epi32(EndClipMask_8x.m256i_u32[0],
                                                               EndClipMask_8x.m256i_u32[1],
                                                               EndClipMask_8x.m256i_u32[2],
                                                               EndClipMask_8x.m256i_u32[3]);
                    
                    __m128i EndClipMask_8xB = _mm_set_epi32(EndClipMask_8x.m256i_u32[4],
                                                               EndClipMask_8x.m256i_u32[5],
                                                               EndClipMask_8x.m256i_u32[6],
                                                               EndClipMask_8x.m256i_u32[7]);
                    
                    StartClipMask_8xA = _mm_and_si128(StartClipMask_8xA, EndClipMask_8xA);
                    StartClipMask_8xB = _mm_and_si128(StartClipMask_8xB, EndClipMask_8xB);
                    
                    StartClipMask_8x = _mm256_set_epi32(StartClipMask_8xB.m128i_u32[0],
                                                         StartClipMask_8xB.m128i_u32[1],
                                                         StartClipMask_8xB.m128i_u32[2],
                                                         StartClipMask_8xB.m128i_u32[3],
                                                         StartClipMask_8xA.m128i_u32[0],
                                                         StartClipMask_8xA.m128i_u32[1],
                                                         StartClipMask_8xA.m128i_u32[2],
                                                         StartClipMask_8xA.m128i_u32[3]);
                    
                }
                #endif
                
                r32 OneOverZIncrement = 0.0f;
                if(XDifference != 0)
                {
                    OneOverZIncrement = (RightOneOverZ-LeftOneOverZ)/((r32)XDifference); 
                }
                
                __m256 CurrentOneOverZ_8x = _mm256_setr_ps(LeftOneOverZ + (XOffset + 0.0f)*OneOverZIncrement,
                                                          LeftOneOverZ + (XOffset + 1.0f)*OneOverZIncrement,
                                                          LeftOneOverZ + (XOffset + 2.0f)*OneOverZIncrement,
                                                          LeftOneOverZ + (XOffset + 3.0f)*OneOverZIncrement,
                                                          LeftOneOverZ + (XOffset + 4.0f)*OneOverZIncrement,
                                                          LeftOneOverZ + (XOffset + 5.0f)*OneOverZIncrement,
                                                          LeftOneOverZ + (XOffset + 6.0f)*OneOverZIncrement,
                                                          LeftOneOverZ + (XOffset + 7.0f)*OneOverZIncrement);
                
                
                __m256 OneOverZIncrement_8x = _mm256_set1_ps(OneOverZIncrement*8.0f);
                
                v2 UVIncrement = {};
                if(XDifference != 0)
                {
                    UVIncrement = 
                    {
                        (RightUV.u-LeftUV.u)/((r32)XDifference),
                        (RightUV.v-LeftUV.v)/((r32)XDifference),
                    };
                }
                
                
                __m256 U_8x = _mm256_setr_ps(LeftUV.u + (XOffset + 0.0f)*UVIncrement.u,
                                            LeftUV.u + (XOffset + 1.0f)*UVIncrement.u,
                                            LeftUV.u + (XOffset + 2.0f)*UVIncrement.u,
                                            LeftUV.u + (XOffset + 3.0f)*UVIncrement.u,
                                            LeftUV.u + (XOffset + 4.0f)*UVIncrement.u,
                                            LeftUV.u + (XOffset + 5.0f)*UVIncrement.u,
                                            LeftUV.u + (XOffset + 6.0f)*UVIncrement.u,
                                            LeftUV.u + (XOffset + 7.0f)*UVIncrement.u);
                
                __m256 V_8x = _mm256_setr_ps(LeftUV.v + (XOffset + 0.0f)*UVIncrement.v,
                                            LeftUV.v + (XOffset + 1.0f)*UVIncrement.v,
                                            LeftUV.v + (XOffset + 2.0f)*UVIncrement.v,
                                            LeftUV.v + (XOffset + 3.0f)*UVIncrement.v,
                                            LeftUV.v + (XOffset + 4.0f)*UVIncrement.v,
                                            LeftUV.v + (XOffset + 5.0f)*UVIncrement.v,
                                            LeftUV.v + (XOffset + 6.0f)*UVIncrement.v,
                                            LeftUV.v + (XOffset + 7.0f)*UVIncrement.v);
                
                __m256 UIncrement_8x = _mm256_set1_ps(UVIncrement.u*8.0f);
                __m256 VIncrement_8x = _mm256_set1_ps(UVIncrement.v*8.0f);
                
                v3 NormalIncrement = {};
                if(XDifference != 0)
                {
                    NormalIncrement = 
                    {
                        (RightNormal.x-LeftNormal.x)/((r32)XDifference),
                        (RightNormal.y-LeftNormal.y)/((r32)XDifference),
                        (RightNormal.z-LeftNormal.z)/((r32)XDifference),
                    };
                }
                
                __m256 NormalX_8x = _mm256_setr_ps(LeftNormal.x + (XOffset + 0.0f)*NormalIncrement.x,
                                                  LeftNormal.x + (XOffset + 1.0f)*NormalIncrement.x,
                                                  LeftNormal.x + (XOffset + 2.0f)*NormalIncrement.x,
                                                  LeftNormal.x + (XOffset + 3.0f)*NormalIncrement.x,
                                                  LeftNormal.x + (XOffset + 4.0f)*NormalIncrement.x,
                                                  LeftNormal.x + (XOffset + 5.0f)*NormalIncrement.x,
                                                  LeftNormal.x + (XOffset + 6.0f)*NormalIncrement.x,
                                                  LeftNormal.x + (XOffset + 7.0f)*NormalIncrement.x);
                
                __m256 NormalY_8x = _mm256_setr_ps(LeftNormal.y + (XOffset + 0.0f)*NormalIncrement.y,
                                                  LeftNormal.y + (XOffset + 1.0f)*NormalIncrement.y,
                                                  LeftNormal.y + (XOffset + 2.0f)*NormalIncrement.y,
                                                  LeftNormal.y + (XOffset + 3.0f)*NormalIncrement.y,
                                                  LeftNormal.y + (XOffset + 4.0f)*NormalIncrement.y,
                                                  LeftNormal.y + (XOffset + 5.0f)*NormalIncrement.y,
                                                  LeftNormal.y + (XOffset + 6.0f)*NormalIncrement.y,
                                                  LeftNormal.y + (XOffset + 7.0f)*NormalIncrement.y);
                
                __m256 NormalZ_8x = _mm256_setr_ps(LeftNormal.z + (XOffset + 0.0f)*NormalIncrement.z,
                                                  LeftNormal.z + (XOffset + 1.0f)*NormalIncrement.z,
                                                  LeftNormal.z + (XOffset + 2.0f)*NormalIncrement.z,
                                                  LeftNormal.z + (XOffset + 3.0f)*NormalIncrement.z,
                                                  LeftNormal.z + (XOffset + 4.0f)*NormalIncrement.z,
                                                  LeftNormal.z + (XOffset + 5.0f)*NormalIncrement.z,
                                                  LeftNormal.z + (XOffset + 6.0f)*NormalIncrement.z,
                                                  LeftNormal.z + (XOffset + 7.0f)*NormalIncrement.z);
                
                unproject_simd_result CurrentNormal = NormalizeVector_8x(NormalX_8x, NormalY_8x, NormalZ_8x);
                
                NormalX_8x = CurrentNormal.VertexX;
                NormalY_8x = CurrentNormal.VertexY;
                NormalZ_8x = CurrentNormal.VertexZ;
                
                __m256 NormalXIncrement_8x = _mm256_set1_ps(NormalIncrement.x*8.0f);
                __m256 NormalYIncrement_8x = _mm256_set1_ps(NormalIncrement.y*8.0f);
                __m256 NormalZIncrement_8x = _mm256_set1_ps(NormalIncrement.z*8.0f);
                
                
                v4 ColorIncrement = {};
                if(XDifference != 0)
                {
                    ColorIncrement = 
                    {
                        (RightColor.r-LeftColor.r)/((r32)XDifference),
                        (RightColor.g-LeftColor.g)/((r32)XDifference),
                        (RightColor.b-LeftColor.b)/((r32)XDifference),
                        (RightColor.a-LeftColor.a)/((r32)XDifference),
                    };
                }
                
                __m256 ColorR_8x = _mm256_set_ps(LeftColor.r + (XOffset + 0.0f)*ColorIncrement.r,
                                                 LeftColor.r + (XOffset + 1.0f)*ColorIncrement.r,
                                                 LeftColor.r + (XOffset + 2.0f)*ColorIncrement.r,
                                                 LeftColor.r + (XOffset + 3.0f)*ColorIncrement.r,
                                                 LeftColor.r + (XOffset + 4.0f)*ColorIncrement.r,
                                                 LeftColor.r + (XOffset + 5.0f)*ColorIncrement.r,
                                                 LeftColor.r + (XOffset + 6.0f)*ColorIncrement.r,
                                                 LeftColor.r + (XOffset + 7.0f)*ColorIncrement.r);
                
                __m256 ColorG_8x = _mm256_set_ps(LeftColor.g + (XOffset + 0.0f)*ColorIncrement.g,
                                                 LeftColor.g + (XOffset + 1.0f)*ColorIncrement.g,
                                                 LeftColor.g + (XOffset + 2.0f)*ColorIncrement.g,
                                                 LeftColor.g + (XOffset + 3.0f)*ColorIncrement.g,
                                                 LeftColor.g + (XOffset + 4.0f)*ColorIncrement.g,
                                                 LeftColor.g + (XOffset + 5.0f)*ColorIncrement.g,
                                                 LeftColor.g + (XOffset + 6.0f)*ColorIncrement.g,
                                                 LeftColor.g + (XOffset + 7.0f)*ColorIncrement.g);
                
                __m256 ColorB_8x = _mm256_set_ps(LeftColor.b + (XOffset + 0.0f)*ColorIncrement.b,
                                                 LeftColor.b + (XOffset + 1.0f)*ColorIncrement.b,
                                                 LeftColor.b + (XOffset + 2.0f)*ColorIncrement.b,
                                                 LeftColor.b + (XOffset + 3.0f)*ColorIncrement.b,
                                                 LeftColor.b + (XOffset + 4.0f)*ColorIncrement.b,
                                                 LeftColor.b + (XOffset + 5.0f)*ColorIncrement.b,
                                                 LeftColor.b + (XOffset + 6.0f)*ColorIncrement.b,
                                                 LeftColor.b + (XOffset + 7.0f)*ColorIncrement.b);
                
                __m256 ColorA_8x = _mm256_set_ps(LeftColor.a + (XOffset + 0.0f)*ColorIncrement.a,
                                                 LeftColor.a + (XOffset + 1.0f)*ColorIncrement.a,
                                                 LeftColor.a + (XOffset + 2.0f)*ColorIncrement.a,
                                                 LeftColor.a + (XOffset + 3.0f)*ColorIncrement.a,
                                                 LeftColor.a + (XOffset + 4.0f)*ColorIncrement.a,
                                                 LeftColor.a + (XOffset + 5.0f)*ColorIncrement.a,
                                                 LeftColor.a + (XOffset + 6.0f)*ColorIncrement.a,
                                                 LeftColor.a + (XOffset + 7.0f)*ColorIncrement.a);
                
                __m256 ColorRIncrement_8x = _mm256_set1_ps(ColorIncrement.r*8.0f);
                __m256 ColorGIncrement_8x = _mm256_set1_ps(ColorIncrement.g*8.0f);
                __m256 ColorBIncrement_8x = _mm256_set1_ps(ColorIncrement.b*8.0f);
                __m256 ColorAIncrement_8x = _mm256_set1_ps(ColorIncrement.a*8.0f);
                
                r32 CurrentZ = CurrentEdgeInList->ZMin;
                r32 ZIncrement = 0.0f;
                
                if(XDifference != 0)
                {
                    ZIncrement = (NextEdgeInList->ZMin-CurrentEdgeInList->ZMin)/((r32)XDifference);
                }
                
                __m256 CurrentZ_8x = _mm256_setr_ps(CurrentEdgeInList->ZMin + (XOffset + 0.0f)*ZIncrement,
                                                    CurrentEdgeInList->ZMin + (XOffset + 1.0f)*ZIncrement,
                                                    CurrentEdgeInList->ZMin + (XOffset + 2.0f)*ZIncrement,
                                                    CurrentEdgeInList->ZMin + (XOffset + 3.0f)*ZIncrement,
                                                    CurrentEdgeInList->ZMin + (XOffset + 4.0f)*ZIncrement,
                                                    CurrentEdgeInList->ZMin + (XOffset + 5.0f)*ZIncrement,
                                                    CurrentEdgeInList->ZMin + (XOffset + 6.0f)*ZIncrement,
                                                    CurrentEdgeInList->ZMin + (XOffset + 7.0f)*ZIncrement);
                
                __m256 ZIncrement_8x = _mm256_set1_ps(8.0f*ZIncrement);
                
                u8 *Row = (u8 *)Buffer->Memory +
                    (s32)LeftX*BITMAP_BYTES_PER_PIXEL +
                    RowIndex*Buffer->Pitch;
                
                r32 *ZBufferPixel = (r32 *)ZBuffer +
                    (s32)LeftX + RowIndex*ZBufferWidth;
                
                u32 *Pixel = (u32 *)Row;
                __m256i ClipMask_8x = StartClipMask_8x;
                
                
                ClipMask_8x = _mm256_set_epi32(ClipMask_8x.m256i_u32[0],
                                               ClipMask_8x.m256i_u32[1],
                                               ClipMask_8x.m256i_u32[2],
                                               ClipMask_8x.m256i_u32[3],
                                               ClipMask_8x.m256i_u32[4],
                                               ClipMask_8x.m256i_u32[5],
                                               ClipMask_8x.m256i_u32[6],
                                               ClipMask_8x.m256i_u32[7]);
                
                for(s32 X = (s32)LeftX;
                    X < (s32)RightX;
                    X += 8)
                {
                    
                    __m256i OriginalDest = _mm256_load_si256((__m256i *)Pixel);
                    
                    void *TextureMemory = Bitmap->Memory;
                    
                    __m256 FinalU_8x = _mm256_mul_ps(_mm256_div_ps(One_8x, CurrentOneOverZ_8x), U_8x);
                    __m256 FinalV_8x = _mm256_mul_ps(_mm256_div_ps(One_8x, CurrentOneOverZ_8x), V_8x);
                    
                    __m256 TexCoordX_8x = _mm256_set1_ps((r32)Bitmap->Width);
                    __m256 TexCoordY_8x = _mm256_set1_ps((r32)Bitmap->Height);
                    
                    TexCoordX_8x = _mm256_mul_ps(TexCoordX_8x, FinalU_8x);
                    TexCoordY_8x = _mm256_mul_ps(TexCoordY_8x, FinalV_8x);
                    
                    __m256i WriteMask = _mm256_castps_si256(_mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(FinalU_8x, Zero_8x, 13),
                                                                                        _mm256_cmp_ps(FinalU_8x, One_8x, 18)),
                                                                          _mm256_and_ps(_mm256_cmp_ps(FinalV_8x, Zero_8x, 13),
                                                                                        _mm256_cmp_ps(FinalV_8x, One_8x, 18))));
                    
                    WriteMask = _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(WriteMask), _mm256_castsi256_ps(ClipMask_8x))); 
                    
                    //__m256i FetchX_8x = _mm256_cvttps_epi32(TexCoordX_8x); 
                    __m128i FetchX_8xA = _mm_cvttps_epi32(_mm_setr_ps(TexCoordX_8x.m256_f32[0],
                                                                      TexCoordX_8x.m256_f32[1],
                                                                      TexCoordX_8x.m256_f32[2],
                                                                      TexCoordX_8x.m256_f32[3]));
                    
                    __m128i FetchX_8xB = _mm_cvttps_epi32(_mm_setr_ps(TexCoordX_8x.m256_f32[4],
                                                                      TexCoordX_8x.m256_f32[5],
                                                                      TexCoordX_8x.m256_f32[6],
                                                                      TexCoordX_8x.m256_f32[7]));
                    
                    __m256i FetchY_8x = _mm256_cvttps_epi32(TexCoordY_8x); 
                    
                    
                    //FetchX_8x = _mm256_slli_epi32(FetchX_8x, 2);
                    FetchX_8xA = _mm_slli_epi32(FetchX_8xA, 2);
                    FetchX_8xB = _mm_slli_epi32(FetchX_8xB, 2);
                    
                    //FetchY_8x = _mm256_or_si256(_mm256_mullo_epi16(FetchY_8x, TexturePitch_8x),                      
                    //                            _mm256_slli_epi32(_mm256_mulhi_epi16(FetchY_8x, TexturePitch_8x), 16));
                    
                    __m128i FetchY_8xA = _mm_setr_epi32(FetchY_8x.m256i_i32[0],
                                                        FetchY_8x.m256i_i32[1],
                                                        FetchY_8x.m256i_i32[2],
                                                        FetchY_8x.m256i_i32[3]);
                    
                    __m128i FetchY_8xB = _mm_setr_epi32(FetchY_8x.m256i_i32[4],
                                                        FetchY_8x.m256i_i32[5],
                                                        FetchY_8x.m256i_i32[6],
                                                        FetchY_8x.m256i_i32[7]);
                    
                    __m128i TexturePitch_8xA = _mm_set1_epi32(TexturePitch_8x.m256i_u32[0]);
                    
                    __m128i TexturePitch_8xB = _mm_set1_epi32(TexturePitch_8x.m256i_u32[0]);
                    
                    FetchY_8xA = _mm_or_si128(_mm_mullo_epi16(FetchY_8xA, TexturePitch_8xA),                      
                                              _mm_slli_epi32(_mm_mulhi_epi16(FetchY_8xA, TexturePitch_8xA), 16));
                    
                    FetchY_8xB = _mm_or_si128(_mm_mullo_epi16(FetchY_8xB, TexturePitch_8xB),                      
                                              _mm_slli_epi32(_mm_mulhi_epi16(FetchY_8xB, TexturePitch_8xB), 16));
                    
                    //__m256i Fetch_8x = _mm256_add_epi32(FetchX_8x, FetchY_8x);
                    __m128i Fetch_8xA = _mm_add_epi32(FetchX_8xA, FetchY_8xA);
                    __m128i Fetch_8xB = _mm_add_epi32(FetchX_8xB, FetchY_8xB);
                    
                    __m256i Fetch_8x = _mm256_set_epi32(Fetch_8xA.m128i_i32[0],
                                                         Fetch_8xA.m128i_i32[1],
                                                         Fetch_8xA.m128i_i32[2],
                                                         Fetch_8xA.m128i_i32[3],
                                                         Fetch_8xB.m128i_i32[0],
                                                         Fetch_8xB.m128i_i32[1],
                                                         Fetch_8xB.m128i_i32[2],
                                                         Fetch_8xB.m128i_i32[3]);
                    
                    s32 Fetch0 = Fetch_8x.m256i_u32[0];
                    s32 Fetch1 = Fetch_8x.m256i_u32[1];
                    s32 Fetch2 = Fetch_8x.m256i_u32[2];
                    s32 Fetch3 = Fetch_8x.m256i_u32[3];
                    s32 Fetch4 = Fetch_8x.m256i_u32[4];
                    s32 Fetch5 = Fetch_8x.m256i_u32[5];
                    s32 Fetch6 = Fetch_8x.m256i_u32[6];
                    s32 Fetch7 = Fetch_8x.m256i_u32[7];
                    
                    u8 *TexelPtr0 = ((u8 *)TextureMemory) + Fetch0;
                    u8 *TexelPtr1 = ((u8 *)TextureMemory) + Fetch1;
                    u8 *TexelPtr2 = ((u8 *)TextureMemory) + Fetch2;
                    u8 *TexelPtr3 = ((u8 *)TextureMemory) + Fetch3;
                    u8 *TexelPtr4 = ((u8 *)TextureMemory) + Fetch4;
                    u8 *TexelPtr5 = ((u8 *)TextureMemory) + Fetch5;
                    u8 *TexelPtr6 = ((u8 *)TextureMemory) + Fetch6;
                    u8 *TexelPtr7 = ((u8 *)TextureMemory) + Fetch7;
                    
#if 0
                    __m256i Sample = _mm256_setr_epi32(*(u32 *)(TexelPtr0),
                                                       *(u32 *)(TexelPtr1),
                                                       *(u32 *)(TexelPtr2),
                                                       *(u32 *)(TexelPtr3),
                                                       *(u32 *)(TexelPtr4),
                                                       *(u32 *)(TexelPtr5),
                                                       *(u32 *)(TexelPtr6),
                                                       *(u32 *)(TexelPtr7));
#endif
                    
                    __m128i SampleA = _mm_setr_epi32(*(u32 *)(TexelPtr0),
                                                     *(u32 *)(TexelPtr1),
                                                     *(u32 *)(TexelPtr2),
                                                     *(u32 *)(TexelPtr3));
                    
                    __m128i SampleB = _mm_setr_epi32(*(u32 *)(TexelPtr4),
                                                     *(u32 *)(TexelPtr5),
                                                     *(u32 *)(TexelPtr6),
                                                     *(u32 *)(TexelPtr7));
                    
                    __m128i MaskFF_4x = _mm_set1_epi32(0xFF);
                    
                    __m128i SampleAR = _mm_and_si128(_mm_srli_epi32(SampleA, 24), MaskFF_4x);
                    __m128i SampleBR = _mm_and_si128(_mm_srli_epi32(SampleB, 24), MaskFF_4x);
                    
                    __m128i SampleAG = _mm_and_si128(_mm_srli_epi32(SampleA, 16), MaskFF_4x);
                    __m128i SampleBG = _mm_and_si128(_mm_srli_epi32(SampleB, 16), MaskFF_4x);
                    
                    __m128i SampleAB = _mm_and_si128(_mm_srli_epi32(SampleA, 8), MaskFF_4x);
                    __m128i SampleBB = _mm_and_si128(_mm_srli_epi32(SampleB, 8), MaskFF_4x);
                    
                    __m128i SampleAA = _mm_and_si128(_mm_srli_epi32(SampleA, 0), MaskFF_4x);
                    __m128i SampleBA = _mm_and_si128(_mm_srli_epi32(SampleB, 0), MaskFF_4x);
                    
                    __m256i SampleFinalR = _mm256_set_epi32(SampleAR.m128i_u32[0],
                                                             SampleAR.m128i_u32[1],
                                                             SampleAR.m128i_u32[2],
                                                             SampleAR.m128i_u32[3],
                                                             SampleBR.m128i_u32[0],
                                                             SampleBR.m128i_u32[1],
                                                             SampleBR.m128i_u32[2],
                                                             SampleBR.m128i_u32[3]);
                    
                    __m256i SampleFinalG = _mm256_set_epi32(SampleAG.m128i_u32[0],
                                                             SampleAG.m128i_u32[1],
                                                             SampleAG.m128i_u32[2],
                                                             SampleAG.m128i_u32[3],
                                                             SampleBG.m128i_u32[0],
                                                             SampleBG.m128i_u32[1],
                                                             SampleBG.m128i_u32[2],
                                                             SampleBG.m128i_u32[3]);
                    
                    __m256i SampleFinalB = _mm256_set_epi32(SampleAB.m128i_u32[0],
                                                             SampleAB.m128i_u32[1],
                                                             SampleAB.m128i_u32[2],
                                                             SampleAB.m128i_u32[3],
                                                             SampleBB.m128i_u32[0],
                                                             SampleBB.m128i_u32[1],
                                                             SampleBB.m128i_u32[2],
                                                             SampleBB.m128i_u32[3]);
                    
                    __m256i SampleFinalA = _mm256_set_epi32(SampleAA.m128i_u32[0],
                                                             SampleAA.m128i_u32[1],
                                                             SampleAA.m128i_u32[2],
                                                             SampleAA.m128i_u32[3],
                                                             SampleBA.m128i_u32[0],
                                                             SampleBA.m128i_u32[1],
                                                             SampleBA.m128i_u32[2],
                                                             SampleBA.m128i_u32[3]);
                    
                    //ColorA_8x = _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(Sample, 24), MaskFF_8x)), One255_8x);
                    //ColorR_8x = _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(Sample, 16), MaskFF_8x)), One255_8x);
                    //ColorG_8x = _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(Sample, 8), MaskFF_8x)), One255_8x);
                    //ColorB_8x = _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(Sample, 0), MaskFF_8x)), One255_8x);
                    
                    ColorA_8x = _mm256_div_ps(_mm256_cvtepi32_ps(SampleFinalR), One255_8x);
                    ColorR_8x = _mm256_div_ps(_mm256_cvtepi32_ps(SampleFinalG), One255_8x);
                    ColorG_8x = _mm256_div_ps(_mm256_cvtepi32_ps(SampleFinalB), One255_8x);
                    ColorB_8x = _mm256_div_ps(_mm256_cvtepi32_ps(SampleFinalA), One255_8x);
                    
                    
                    __m256 FinalColorR_8x = _mm256_set1_ps(0.0f);
                    __m256 FinalColorG_8x = _mm256_set1_ps(0.0f);
                    __m256 FinalColorB_8x = _mm256_set1_ps(0.0f);
                    __m256 FinalColorA_8x = _mm256_set1_ps(0.0f);
                    
                    if(PhongShading)
                    {
                        light_data *Lights = &Commands->LightData; 
                        light_info *Light = Lights->Lights;
                        
                        unproject_simd_result CurrentVertexP = 
                            UnprojectVertex_8x((r32)X, (r32)RowIndex, CurrentZ_8x, &Commands->Transform);
                        
                        for(u32 LightIndex = 0;
                            LightIndex < Lights->LightCount;
                            ++LightIndex, ++Light)
                        {
                            if(LightIndex == 0)
                            {
                                FinalColorR_8x = _mm256_mul_ps(ColorR_8x, _mm256_set1_ps(Lights->AmbientIntensity.r));
                                FinalColorG_8x = _mm256_mul_ps(ColorG_8x, _mm256_set1_ps(Lights->AmbientIntensity.g));
                                FinalColorB_8x = _mm256_mul_ps(ColorB_8x, _mm256_set1_ps(Lights->AmbientIntensity.b));
                                FinalColorA_8x = _mm256_mul_ps(ColorA_8x, _mm256_set1_ps(Lights->AmbientIntensity.a));
                            }
                            
                            __m256 LightX = _mm256_set1_ps(Light->P.x);
                            __m256 LightY = _mm256_set1_ps(Light->P.y);
                            __m256 LightZ = _mm256_set1_ps(Light->P.z);
                            
                            __m256 VectorToLightX_8x = _mm256_sub_ps(LightX, CurrentVertexP.VertexX);
                            __m256 VectorToLightY_8x = _mm256_sub_ps(LightY, CurrentVertexP.VertexY);
                            __m256 VectorToLightZ_8x = _mm256_sub_ps(LightZ, CurrentVertexP.VertexZ);
                            
                            unproject_simd_result VectorToLight = NormalizeVector_8x(VectorToLightX_8x, VectorToLightY_8x, VectorToLightZ_8x);

                            unproject_simd_result LightDirection = VectorToLight;
                            
                            __m256 DotNormalLightX_8x = _mm256_mul_ps(NormalX_8x, VectorToLight.VertexX);
                            __m256 DotNormalLightY_8x = _mm256_mul_ps(NormalY_8x, VectorToLight.VertexY);
                            __m256 DotNormalLightZ_8x = _mm256_mul_ps(NormalZ_8x, VectorToLight.VertexZ);
                            
                            __m256 CosineOfIncidence_8x = _mm256_add_ps(_mm256_add_ps(DotNormalLightX_8x, DotNormalLightY_8x), DotNormalLightZ_8x);
                            
                            CosineOfIncidence_8x = _mm256_min_ps(One_8x, _mm256_max_ps(Zero_8x, CosineOfIncidence_8x));
                            
                            unproject_simd_result ViewDirection = NormalizeVector_8x(
                                _mm256_sub_ps(Zero_8x, CurrentVertexP.VertexX), 
                                _mm256_sub_ps(Zero_8x, CurrentVertexP.VertexY), 
                                _mm256_sub_ps(Zero_8x, CurrentVertexP.VertexZ));
                            
                            __m256 HalfAngleX_8x = _mm256_add_ps(LightDirection.VertexX, ViewDirection.VertexX);
                            __m256 HalfAngleY_8x = _mm256_add_ps(LightDirection.VertexY, ViewDirection.VertexY);
                            __m256 HalfAngleZ_8x = _mm256_add_ps(LightDirection.VertexZ, ViewDirection.VertexZ);
                            
                            unproject_simd_result HalfAngle = NormalizeVector_8x(HalfAngleX_8x, HalfAngleY_8x, HalfAngleZ_8x);
                            
                            __m256 PhongTerm_8x = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(NormalX_8x, HalfAngle.VertexX), _mm256_mul_ps(NormalY_8x, HalfAngle.VertexY)), _mm256_mul_ps(NormalZ_8x, HalfAngle.VertexZ));
                            
                            PhongTerm_8x = _mm256_min_ps(One_8x, _mm256_max_ps(Zero_8x, PhongTerm_8x));
                            
                            for(u32 FactorIndex = 0;
                                FactorIndex < 4;
                                ++FactorIndex)
                            {
                                PhongTerm_8x = _mm256_mul_ps(PhongTerm_8x, PhongTerm_8x);
                            }
                            
                            __m256 SpecularColorR_8x = _mm256_set1_ps(1.0f); 
                            __m256 SpecularColorG_8x = _mm256_set1_ps(1.0f); 
                            __m256 SpecularColorB_8x = _mm256_set1_ps(1.0f); 
                            __m256 SpecularColorA_8x = _mm256_set1_ps(1.0f); 
                            
                            __m256 LightIntensityR_8x = _mm256_set1_ps(Light->Intensity.r); 
                            __m256 LightIntensityG_8x = _mm256_set1_ps(Light->Intensity.g); 
                            __m256 LightIntensityB_8x = _mm256_set1_ps(Light->Intensity.b); 
                            __m256 LightIntensityA_8x = _mm256_set1_ps(Light->Intensity.a); 
                            
                            FinalColorR_8x = _mm256_add_ps(FinalColorR_8x, 
                                                           _mm256_add_ps(_mm256_mul_ps(CosineOfIncidence_8x, _mm256_mul_ps(ColorR_8x, LightIntensityR_8x)), 
                                                                         _mm256_mul_ps(PhongTerm_8x, _mm256_mul_ps(SpecularColorR_8x, LightIntensityR_8x))));
                            
                            FinalColorG_8x = _mm256_add_ps(FinalColorG_8x, 
                                                           _mm256_add_ps(_mm256_mul_ps(CosineOfIncidence_8x, _mm256_mul_ps(ColorG_8x, LightIntensityG_8x)), 
                                                                         _mm256_mul_ps(PhongTerm_8x, _mm256_mul_ps(SpecularColorG_8x, LightIntensityG_8x))));
                            
                            FinalColorB_8x = _mm256_add_ps(FinalColorB_8x, 
                                                           _mm256_add_ps(_mm256_mul_ps(CosineOfIncidence_8x, _mm256_mul_ps(ColorB_8x, LightIntensityB_8x)), 
                                                                         _mm256_mul_ps(PhongTerm_8x, _mm256_mul_ps(SpecularColorB_8x, LightIntensityB_8x))));
                            
                            FinalColorA_8x = _mm256_add_ps(FinalColorA_8x, 
                                                           _mm256_add_ps(_mm256_mul_ps(CosineOfIncidence_8x, _mm256_mul_ps(ColorA_8x, LightIntensityA_8x)),
                                                                         _mm256_mul_ps(PhongTerm_8x, _mm256_mul_ps(SpecularColorA_8x, LightIntensityA_8x))));
                        }
                        
                        FinalColorR_8x = _mm256_max_ps(_mm256_min_ps(FinalColorR_8x, One_8x), Zero_8x);
                        FinalColorG_8x = _mm256_max_ps(_mm256_min_ps(FinalColorG_8x, One_8x), Zero_8x);
                        FinalColorB_8x = _mm256_max_ps(_mm256_min_ps(FinalColorB_8x, One_8x), Zero_8x);
                        FinalColorA_8x = _mm256_max_ps(_mm256_min_ps(FinalColorA_8x, One_8x), Zero_8x);
                        
                        //__m256i Color32_8x = _mm256_or_si256(_mm256_or_si256(_mm256_or_si256(_mm256_srli_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(FinalColorG_8x, One255_8x)), 8), 
                        //                                                                     _mm256_cvtps_epi32(_mm256_mul_ps(FinalColorG_8x, One255_8x))),
                        //                                                     _mm256_srli_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(FinalColorR_8x, One255_8x)), 16)),
                        //                                     _mm256_srli_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(FinalColorA_8x, One255_8x)), 24));
                        
                        __m256i Color32R_8x = _mm256_cvtps_epi32(_mm256_mul_ps(FinalColorR_8x, One255_8x));
                        __m256i Color32G_8x = _mm256_cvtps_epi32(_mm256_mul_ps(FinalColorG_8x, One255_8x));
                        __m256i Color32B_8x = _mm256_cvtps_epi32(_mm256_mul_ps(FinalColorB_8x, One255_8x));
                        __m256i Color32A_8x = _mm256_cvtps_epi32(_mm256_mul_ps(FinalColorA_8x, One255_8x));
                        
                        __m128i Color32R_8xA = _mm_set_epi32(Color32R_8x.m256i_u32[0],
                                                             Color32R_8x.m256i_u32[1],
                                                             Color32R_8x.m256i_u32[2],
                                                             Color32R_8x.m256i_u32[3]);
                        
                        __m128i Color32R_8xB = _mm_set_epi32(Color32R_8x.m256i_u32[4],
                                                             Color32R_8x.m256i_u32[5],
                                                             Color32R_8x.m256i_u32[6],
                                                             Color32R_8x.m256i_u32[7]);
                        
                        __m128i Color32G_8xA = _mm_set_epi32(Color32G_8x.m256i_u32[0],
                                                             Color32G_8x.m256i_u32[1],
                                                             Color32G_8x.m256i_u32[2],
                                                             Color32G_8x.m256i_u32[3]);
                        
                        __m128i Color32G_8xB = _mm_set_epi32(Color32G_8x.m256i_u32[4],
                                                             Color32G_8x.m256i_u32[5],
                                                             Color32G_8x.m256i_u32[6],
                                                             Color32G_8x.m256i_u32[7]);
                        
                        __m128i Color32B_8xA = _mm_set_epi32(Color32B_8x.m256i_u32[0],
                                                             Color32B_8x.m256i_u32[1],
                                                             Color32B_8x.m256i_u32[2],
                                                             Color32B_8x.m256i_u32[3]);
                        
                        __m128i Color32B_8xB = _mm_set_epi32(Color32B_8x.m256i_u32[4],
                                                             Color32B_8x.m256i_u32[5],
                                                             Color32B_8x.m256i_u32[6],
                                                             Color32B_8x.m256i_u32[7]);
                        
                        __m128i Color32A_8xA = _mm_set_epi32(Color32A_8x.m256i_u32[0],
                                                             Color32A_8x.m256i_u32[1],
                                                             Color32A_8x.m256i_u32[2],
                                                             Color32A_8x.m256i_u32[3]);
                        
                        __m128i Color32A_8xB = _mm_set_epi32(Color32A_8x.m256i_u32[4],
                                                             Color32A_8x.m256i_u32[5],
                                                             Color32A_8x.m256i_u32[6],
                                                             Color32A_8x.m256i_u32[7]);
                        
                        Color32R_8xA = _mm_slli_epi32(Color32R_8xA, 16);
                        Color32R_8xB = _mm_slli_epi32(Color32R_8xB, 16);
                        Color32G_8xA = _mm_slli_epi32(Color32G_8xA, 8);
                        Color32G_8xB = _mm_slli_epi32(Color32G_8xB, 8);
                        Color32B_8xA = _mm_slli_epi32(Color32B_8xA, 0);
                        Color32B_8xB = _mm_slli_epi32(Color32B_8xB, 0);
                        Color32A_8xA = _mm_slli_epi32(Color32A_8xA, 24);
                        Color32A_8xB = _mm_slli_epi32(Color32A_8xB, 24);
                        
                        __m128i Color32_8xA = _mm_or_si128(_mm_or_si128(_mm_or_si128(Color32R_8xA, Color32G_8xA), Color32B_8xA), Color32A_8xA);
                        __m128i Color32_8xB = _mm_or_si128(_mm_or_si128(_mm_or_si128(Color32R_8xB, Color32G_8xB), Color32B_8xB), Color32A_8xB);
                        
                        __m256i Color32_8x = _mm256_set_epi32(Color32_8xB.m128i_u32[0],
                                                              Color32_8xB.m128i_u32[1],
                                                              Color32_8xB.m128i_u32[2],
                                                              Color32_8xB.m128i_u32[3],
                                                              Color32_8xA.m128i_u32[0],
                                                              Color32_8xA.m128i_u32[1],
                                                              Color32_8xA.m128i_u32[2],
                                                              Color32_8xA.m128i_u32[3]
                                                              );
                        
                        __m256 OriginalZ_8x = _mm256_load_ps(ZBufferPixel); 
                        #if 0
                        OriginalZ_8x = _mm256_set_ps(OriginalZ_8x.m256_f32[0],
                                                     OriginalZ_8x.m256_f32[1],
                                                     OriginalZ_8x.m256_f32[2],
                                                     OriginalZ_8x.m256_f32[3],
                                                     OriginalZ_8x.m256_f32[4],
                                                     OriginalZ_8x.m256_f32[5],
                                                     OriginalZ_8x.m256_f32[6],
                                                     OriginalZ_8x.m256_f32[7]);
                        #endif
                        __m256 ZMask = _mm256_cmp_ps(CurrentZ_8x, OriginalZ_8x, 29);
                        
                        
                        WriteMask = _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(WriteMask), ZMask));
                        
                        ZMask = _mm256_and_ps(ZMask, _mm256_castsi256_ps(WriteMask));
                        
                        __m256 NewZ_8x = _mm256_or_ps(_mm256_and_ps(ZMask, CurrentZ_8x),
                                                      _mm256_andnot_ps(ZMask, OriginalZ_8x));
                        
                        Color32_8x = _mm256_castps_si256(_mm256_or_ps(_mm256_and_ps(_mm256_castsi256_ps(WriteMask), _mm256_castsi256_ps(Color32_8x)),
                                                                      _mm256_andnot_ps(_mm256_castsi256_ps(WriteMask), _mm256_castsi256_ps(OriginalDest))));
                        
                        if((X + 16) < RightX)
                        {                  
                            ClipMask_8x = _mm256_set1_epi8(-1);
                        }
                        else
                        {                  
                            ClipMask_8x = EndClipMask_8x;
                            ClipMask_8x = _mm256_set_epi32(ClipMask_8x.m256i_u32[0],
                                                           ClipMask_8x.m256i_u32[1],
                                                           ClipMask_8x.m256i_u32[2],
                                                           ClipMask_8x.m256i_u32[3],
                                                           ClipMask_8x.m256i_u32[4],
                                                           ClipMask_8x.m256i_u32[5],
                                                           ClipMask_8x.m256i_u32[6],
                                                           ClipMask_8x.m256i_u32[7]);
                        }
                        
                        _mm256_store_ps(ZBufferPixel, NewZ_8x);
                        _mm256_store_si256((__m256i *)Pixel, Color32_8x);
                        
                        ZBufferPixel += 8;
                        Pixel += 8;
                        
                        unproject_simd_result NewNormals = NormalizeVector_8x(
                            _mm256_add_ps(NormalX_8x, NormalXIncrement_8x),
                            _mm256_add_ps(NormalY_8x, NormalYIncrement_8x),
                            _mm256_add_ps(NormalZ_8x, NormalZIncrement_8x)
                            );
                        
                        NormalX_8x = NewNormals.VertexX;
                        NormalY_8x = NewNormals.VertexY;
                        NormalZ_8x = NewNormals.VertexZ;
                        
                        ColorR_8x = _mm256_add_ps(ColorR_8x, ColorRIncrement_8x);
                        ColorG_8x = _mm256_add_ps(ColorG_8x, ColorGIncrement_8x);
                        ColorB_8x = _mm256_add_ps(ColorB_8x, ColorBIncrement_8x);
                        ColorA_8x = _mm256_add_ps(ColorA_8x, ColorAIncrement_8x);
                        
                        CurrentZ_8x = _mm256_add_ps(CurrentZ_8x, ZIncrement_8x);
                        
                        CurrentOneOverZ_8x = _mm256_add_ps(CurrentOneOverZ_8x, OneOverZIncrement_8x);
                        
                        U_8x = _mm256_add_ps(U_8x, UIncrement_8x);
                        V_8x = _mm256_add_ps(V_8x, VIncrement_8x);
                        
                    }
                    else
                    {
                        
                        
                        FinalColorR_8x = ColorR_8x;
                        FinalColorR_8x = ColorG_8x;
                        FinalColorR_8x = ColorB_8x;
                        FinalColorR_8x = ColorA_8x;
                        
                        __m256i Color32_8x = _mm256_or_si256(_mm256_or_si256(_mm256_or_si256(_mm256_srli_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(FinalColorG_8x, One255_8x)), 8), 
                                                                                             _mm256_cvtps_epi32(_mm256_mul_ps(FinalColorG_8x, One255_8x))),
                                                                             _mm256_srli_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(FinalColorR_8x, One255_8x)), 16)),
                                                             _mm256_srli_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(FinalColorA_8x, One255_8x)), 24));
                        
                        if(CurrentZ > *ZBufferPixel)
                        {
                            _mm256_store_ps(ZBufferPixel, CurrentZ_8x);
                            _mm256_store_si256((__m256i *)Pixel, Color32_8x);
                        }
                        ZBufferPixel += 8;
                        Pixel += 8;
                        
                        
                        
                        ColorR_8x = _mm256_add_ps(ColorR_8x, ColorRIncrement_8x);
                        ColorG_8x = _mm256_add_ps(ColorG_8x, ColorGIncrement_8x);
                        ColorB_8x = _mm256_add_ps(ColorB_8x, ColorBIncrement_8x);
                        ColorA_8x = _mm256_add_ps(ColorA_8x, ColorAIncrement_8x);
                        
                        CurrentZ_8x = _mm256_add_ps(CurrentZ_8x, ZIncrement_8x);
                        
                    }
                }
            }
            
            CurrentEdgeInList->XMin += CurrentEdgeInList->Gradient;
            NextEdgeInList->XMin += NextEdgeInList->Gradient;
            
            CurrentEdgeInList->ZMin += CurrentEdgeInList->ZGradient;
            NextEdgeInList->ZMin += NextEdgeInList->ZGradient;
            
            CurrentEdgeInList->MinColor += CurrentEdgeInList->ColorGradient;
            NextEdgeInList->MinColor += NextEdgeInList->ColorGradient;
            
            CurrentEdgeInList->MinNormal = Normalize(CurrentEdgeInList->MinNormal + CurrentEdgeInList->NormalGradient);
            NextEdgeInList->MinNormal = Normalize(NextEdgeInList->MinNormal + NextEdgeInList->NormalGradient);
            
            CurrentEdgeInList->UMin += CurrentEdgeInList->UGradient;
            CurrentEdgeInList->VMin += CurrentEdgeInList->VGradient;
            CurrentEdgeInList->OneOverZMin += CurrentEdgeInList->OneOverZGradient;

            NextEdgeInList->UMin += NextEdgeInList->UGradient;
            NextEdgeInList->VMin += NextEdgeInList->VGradient;
            NextEdgeInList->OneOverZMin += NextEdgeInList->OneOverZGradient;
            
            if(CurrentEdgeInList->XMin > NextEdgeInList->XMin)
            {
                CurrentEdgeInList->Next = NextEdgeInList->Next;
                NextEdgeInList->Next = CurrentEdgeInList;
                if(PreviousNextEdgeInList)
                {
                    PreviousNextEdgeInList->Next = NextEdgeInList;
                }
                CurrentEdgeInList = NextEdgeInList;
                NextEdgeInList = CurrentEdgeInList->Next;
            }
            
            if(PreviousNextEdgeInList)
            {
                if(PreviousNextEdgeInList->XMin > CurrentEdgeInList->XMin)
                {
                    PreviousNextEdgeInList->Next = CurrentEdgeInList->Next;
                    CurrentEdgeInList->Next = PreviousNextEdgeInList;
                    PreviousCurrentEdgeInList->Next = CurrentEdgeInList;
                    PreviousNextEdgeInList = CurrentEdgeInList;
                    CurrentEdgeInList = PreviousNextEdgeInList->Next;
                }
            }
            
            PreviousCurrentEdgeInList = CurrentEdgeInList;
            PreviousNextEdgeInList = NextEdgeInList;
            
            if(NextEdgeInList->Next)
            {
                CurrentEdgeInList = NextEdgeInList->Next;
                NextEdgeInList = CurrentEdgeInList->Next;
            }
            else
            {
                NextEdgeInList = 0;
            }
        }
    }

}

global_variable u32 WorkIndex = 0;

internal void
DrawModelOptimizedLines(platform_work_queue *RenderQueue,
                           loaded_bitmap *Buffer,
                           edge_info *Edges, u32 EdgeCount,
                           game_render_commands *Commands,
                           loaded_bitmap *Bitmap = 0,
                           b32 PhongShading = 0)
{
    r32 *ZBuffer = Commands->ZBuffer;
    u32 ZBufferWidth = Commands->Width;
    
    s32 FirstRow = Edges[0].YMin;
    
    s32 Height;
    s32 MaxRow = Edges[0].YMax;
    for(u32 EdgeIndex = 1;
        EdgeIndex < EdgeCount;
        ++EdgeIndex)
    {
        if(MaxRow < Edges[EdgeIndex].YMax)
        {
            MaxRow = Edges[EdgeIndex].YMax;
        }
    }
    
    Height = MaxRow - FirstRow;

    edge_info *ListHead = 0;
    edge_info *ListTail = 0;
    
    s32 MaxY = FirstRow + Height;
    if(MaxY > Buffer->Height)
    {
        MaxY = Buffer->Height;
    }
    
    for(s32 RowIndex = FirstRow;
        RowIndex < MaxY;
        ++RowIndex)
    {
        
        thread_edge_info ThreadEdges[1000] = {};
        u32 ThreadEdgeIndex = 0;
        
        for(u32 EdgeIndex = 0;
            EdgeIndex < EdgeCount;
            ++EdgeIndex)
        {
            
            edge_info *CurrentEdge = Edges + EdgeIndex;
            
            if(CurrentEdge->YMin == RowIndex)
            {
                if(ListHead)
                {
                    if(CurrentEdge->XMin < ListHead->XMin || 
                       (CurrentEdge->XMin == ListHead->XMin &&
                        (CurrentEdge->Gradient < ListHead->Gradient ||
                         (CurrentEdge->Gradient == ListHead->Gradient &&
                          CurrentEdge->Left < ListHead->Left))))
                    {
                        CurrentEdge->Next = ListHead;
                        ListHead = CurrentEdge;
                    }
                    else
                    {
                        edge_info *ComparedEdge = ListHead;
                        edge_info *PreviousEdge = ListHead;
                        while(ComparedEdge != ListTail)
                        {
                            ComparedEdge = ComparedEdge->Next;
                            
                            if(CurrentEdge->XMin < ComparedEdge->XMin || 
                               (CurrentEdge->XMin == ComparedEdge->XMin &&
                                (CurrentEdge->Gradient < ComparedEdge->Gradient ||
                                 (CurrentEdge->Gradient == ComparedEdge->Gradient &&
                                  CurrentEdge->Left < ComparedEdge->Left))))
                            {
                                CurrentEdge->Next = ComparedEdge;
                                PreviousEdge->Next = CurrentEdge;
                                ComparedEdge = ListTail;
                            }
                            else
                            {
                                PreviousEdge = ComparedEdge;
                            }
                            
                        }                        
                        
                        if(PreviousEdge == ComparedEdge)
                        {
                            ListTail->Next = CurrentEdge;
                            ListTail = CurrentEdge;
                        }
                    }

                }
                else
                {
                    ListHead = CurrentEdge;
                    ListTail = ListHead;
                }
            }
        }
        
        while(ListHead->YMax <= RowIndex)
        {
            edge_info *RemovedEdge = ListHead;
            ListHead = ListHead->Next;
            RemovedEdge->Next = 0;
        }
        
        edge_info *PreviousEdge = ListHead;
        edge_info *CheckedEdge = ListHead;
        
        if(CheckedEdge)
        {
            while(CheckedEdge != ListTail)
            {
                CheckedEdge = CheckedEdge->Next;
                
                if(CheckedEdge->YMax <= RowIndex)
                {
                    
                    if(CheckedEdge == ListTail)
                    {
                        ListTail = PreviousEdge;
                        ListTail->Next = 0;
                        CheckedEdge = ListTail;
                    }
                    else
                    {
                        PreviousEdge->Next = CheckedEdge->Next;
                        CheckedEdge = PreviousEdge;
                    }
                }
                
                PreviousEdge = CheckedEdge;
            }
        }
        
        edge_info *PreviousCurrentEdgeInList = 0;
        edge_info *PreviousNextEdgeInList = 0;
        edge_info *CurrentEdgeInList = ListHead;
        edge_info *NextEdgeInList = CurrentEdgeInList->Next;
        
        
        buffer_line_render_work Work = {};
        
        Work.Commands = Commands;
        Work.OutputTarget = Buffer;
        Work.Bitmap = Bitmap;
        Work.RowIndex = RowIndex;
        Work.PhongShading = PhongShading;
        Work.EdgeCount = 0;
        
        while(NextEdgeInList != 0)
        {
            
            Work.EdgeCount++;
            
            ThreadEdges[ThreadEdgeIndex].LeftXMin = CurrentEdgeInList->XMin;
            ThreadEdges[ThreadEdgeIndex].RightXMin = NextEdgeInList->XMin;
            
            ThreadEdges[ThreadEdgeIndex].LeftZMin = CurrentEdgeInList->ZMin;
            ThreadEdges[ThreadEdgeIndex].RightZMin = NextEdgeInList->ZMin;
            
            ThreadEdges[ThreadEdgeIndex].LeftOneOverZMin = CurrentEdgeInList->OneOverZMin;
            ThreadEdges[ThreadEdgeIndex].RightOneOverZMin = NextEdgeInList->OneOverZMin;
            
            ThreadEdges[ThreadEdgeIndex].LeftUMin = CurrentEdgeInList->UMin;
            ThreadEdges[ThreadEdgeIndex].RightUMin = NextEdgeInList->UMin;
            
            ThreadEdges[ThreadEdgeIndex].LeftVMin = CurrentEdgeInList->VMin;
            ThreadEdges[ThreadEdgeIndex].RightVMin = NextEdgeInList->VMin;
            
            ThreadEdges[ThreadEdgeIndex].LeftMinColor = CurrentEdgeInList->MinColor;
            ThreadEdges[ThreadEdgeIndex].RightMinColor = NextEdgeInList->MinColor;
            
            ThreadEdges[ThreadEdgeIndex].LeftMinNormal = CurrentEdgeInList->MinNormal;
            ThreadEdges[ThreadEdgeIndex].RightMinNormal = NextEdgeInList->MinNormal;
            
            ThreadEdgeIndex++;
            
            CurrentEdgeInList->XMin += CurrentEdgeInList->Gradient;
            NextEdgeInList->XMin += NextEdgeInList->Gradient;
            
            CurrentEdgeInList->ZMin += CurrentEdgeInList->ZGradient;
            NextEdgeInList->ZMin += NextEdgeInList->ZGradient;
            
            CurrentEdgeInList->MinColor += CurrentEdgeInList->ColorGradient;
            NextEdgeInList->MinColor += NextEdgeInList->ColorGradient;
            
            CurrentEdgeInList->MinNormal = Normalize(CurrentEdgeInList->MinNormal + CurrentEdgeInList->NormalGradient);
            NextEdgeInList->MinNormal = Normalize(NextEdgeInList->MinNormal + NextEdgeInList->NormalGradient);
            
            CurrentEdgeInList->UMin += CurrentEdgeInList->UGradient;
            CurrentEdgeInList->VMin += CurrentEdgeInList->VGradient;
            CurrentEdgeInList->OneOverZMin += CurrentEdgeInList->OneOverZGradient;
            
            NextEdgeInList->UMin += NextEdgeInList->UGradient;
            NextEdgeInList->VMin += NextEdgeInList->VGradient;
            NextEdgeInList->OneOverZMin += NextEdgeInList->OneOverZGradient;
            
            if(CurrentEdgeInList->XMin > NextEdgeInList->XMin)
            {
                CurrentEdgeInList->Next = NextEdgeInList->Next;
                NextEdgeInList->Next = CurrentEdgeInList;
                if(PreviousNextEdgeInList)
                {
                    PreviousNextEdgeInList->Next = NextEdgeInList;
                }
                CurrentEdgeInList = NextEdgeInList;
                NextEdgeInList = CurrentEdgeInList->Next;
            }
            
            if(PreviousNextEdgeInList)
            {
                if(PreviousNextEdgeInList->XMin > CurrentEdgeInList->XMin)
                {
                    PreviousNextEdgeInList->Next = CurrentEdgeInList->Next;
                    CurrentEdgeInList->Next = PreviousNextEdgeInList;
                    PreviousCurrentEdgeInList->Next = CurrentEdgeInList;
                    PreviousNextEdgeInList = CurrentEdgeInList;
                    CurrentEdgeInList = PreviousNextEdgeInList->Next;
                }
            }
            
            PreviousCurrentEdgeInList = CurrentEdgeInList;
            PreviousNextEdgeInList = NextEdgeInList;
            
            if(NextEdgeInList->Next)
            {
                CurrentEdgeInList = NextEdgeInList->Next;
                NextEdgeInList = CurrentEdgeInList->Next;
            }
            else
            {
                NextEdgeInList = 0;
            }
        }
        
        void *Data = AddWorkToThreadMemory(Commands, &Work, sizeof(buffer_line_render_work) - sizeof(thread_edge_info));
        //buffer_line_render_work *EdgeData = (buffer_line_render_work *)Data;
        //EdgeData->Edges = (thread_edge_info *)AddWorkToThreadMemory(Commands, ThreadEdges, ThreadEdgeIndex*sizeof(thread_edge_info));
        AddWorkToThreadMemory(Commands, ThreadEdges, ThreadEdgeIndex*sizeof(thread_edge_info));
        
        Platform.AddEntry(RenderQueue, DoBufferLineRenderWork, Data);
        
    }
    
}

internal void
DrawModelOptimized(platform_work_queue *RenderQueue,
                      loaded_bitmap *Buffer,
                      edge_info *Edges, u32 EdgeCount,
                      game_render_commands *Commands,
                      loaded_bitmap *Bitmap = 0,
                      b32 PhongShading = 0)
{
    r32 *ZBuffer = Commands->ZBuffer;
    u32 ZBufferWidth = Commands->Width;
    
    s32 FirstRow = Edges[0].YMin;
    
    s32 Height;
    s32 MaxRow = Edges[0].YMax;
    for(u32 EdgeIndex = 1;
        EdgeIndex < EdgeCount;
        ++EdgeIndex)
    {
        if(MaxRow < Edges[EdgeIndex].YMax)
        {
            MaxRow = Edges[EdgeIndex].YMax;
        }
    }
    
    Height = MaxRow - FirstRow;

    edge_info *ListHead = 0;
    edge_info *ListTail = 0;
    
    s32 MaxY = FirstRow + Height;
    if(MaxY > Buffer->Height)
    {
        MaxY = Buffer->Height;
    }
    for(s32 RowIndex = FirstRow;
        RowIndex < MaxY;
        ++RowIndex)
    {
        for(u32 EdgeIndex = 0;
            EdgeIndex < EdgeCount;
            ++EdgeIndex)
        {
            
            edge_info *CurrentEdge = Edges + EdgeIndex;
            
            if(CurrentEdge->YMin == RowIndex)
            {
                if(ListHead)
                {
                    if(CurrentEdge->XMin < ListHead->XMin || 
                       (CurrentEdge->XMin == ListHead->XMin &&
                        (CurrentEdge->Gradient < ListHead->Gradient ||
                         (CurrentEdge->Gradient == ListHead->Gradient &&
                          CurrentEdge->Left < ListHead->Left))))
                    {
                        CurrentEdge->Next = ListHead;
                        ListHead = CurrentEdge;
                    }
                    else
                    {
                        edge_info *ComparedEdge = ListHead;
                        edge_info *PreviousEdge = ListHead;
                        while(ComparedEdge != ListTail)
                        {
                            ComparedEdge = ComparedEdge->Next;
                            
                            if(CurrentEdge->XMin < ComparedEdge->XMin || 
                               (CurrentEdge->XMin == ComparedEdge->XMin &&
                                (CurrentEdge->Gradient < ComparedEdge->Gradient ||
                                 (CurrentEdge->Gradient == ComparedEdge->Gradient &&
                                  CurrentEdge->Left < ComparedEdge->Left))))
                            {
                                CurrentEdge->Next = ComparedEdge;
                                PreviousEdge->Next = CurrentEdge;
                                ComparedEdge = ListTail;
                            }
                            else
                            {
                                PreviousEdge = ComparedEdge;
                            }
                            
                        }                        
                        
                        if(PreviousEdge == ComparedEdge)
                        {
                            ListTail->Next = CurrentEdge;
                            ListTail = CurrentEdge;
                        }
                    }

                }
                else
                {
                    ListHead = CurrentEdge;
                    ListTail = ListHead;
                }
            }
        }
        
        while(ListHead->YMax <= RowIndex)
        {
            edge_info *RemovedEdge = ListHead;
            ListHead = ListHead->Next;
            RemovedEdge->Next = 0;
        }
        
        edge_info *PreviousEdge = ListHead;
        edge_info *CheckedEdge = ListHead;
        
        if(CheckedEdge)
        {
            while(CheckedEdge != ListTail)
            {
                CheckedEdge = CheckedEdge->Next;
                
                if(CheckedEdge->YMax <= RowIndex)
                {
                    
                    if(CheckedEdge == ListTail)
                    {
                        ListTail = PreviousEdge;
                        ListTail->Next = 0;
                        CheckedEdge = ListTail;
                    }
                    else
                    {
                        PreviousEdge->Next = CheckedEdge->Next;
                        CheckedEdge = PreviousEdge;
                    }
                }
                
                PreviousEdge = CheckedEdge;
            }
        }
        
        edge_info *PreviousCurrentEdgeInList = 0;
        edge_info *PreviousNextEdgeInList = 0;
        edge_info *CurrentEdgeInList = ListHead;
        edge_info *NextEdgeInList = CurrentEdgeInList->Next;
        
        line_render_work *WorkArray = (line_render_work *)Commands->ThreadMemory;
        while(NextEdgeInList != 0)
        {
            line_render_work *Work = WorkArray + WorkIndex++;
            
            edge_info CurrentEdge;
            CurrentEdge.YMax = CurrentEdgeInList->YMax;
            CurrentEdge.XMin = CurrentEdgeInList->XMin;
            CurrentEdge.ZMin = CurrentEdgeInList->ZMin;
            CurrentEdge.OneOverZMin = CurrentEdgeInList->OneOverZMin;
            CurrentEdge.Gradient = CurrentEdgeInList->Gradient;
            CurrentEdge.ZGradient = CurrentEdgeInList->ZGradient;
            CurrentEdge.OneOverZGradient = CurrentEdgeInList->OneOverZGradient;
            CurrentEdge.YMin = CurrentEdgeInList->YMin;
            CurrentEdge.UMin = CurrentEdgeInList->UMin;
            CurrentEdge.VMin = CurrentEdgeInList->VMin;
            CurrentEdge.UGradient = CurrentEdgeInList->UGradient;
            CurrentEdge.VGradient = CurrentEdgeInList->VGradient;
            CurrentEdge.Left = CurrentEdgeInList->Left;
            CurrentEdge.MinColor = CurrentEdgeInList->MinColor;
            CurrentEdge.ColorGradient = CurrentEdgeInList->ColorGradient;
            CurrentEdge.MinNormal = CurrentEdgeInList->MinNormal;
            CurrentEdge.NormalGradient = CurrentEdgeInList->NormalGradient;
            CurrentEdge.Next = 0;
            
            edge_info NextEdge;
            NextEdge.YMax = NextEdgeInList->YMax;
            NextEdge.XMin = NextEdgeInList->XMin;
            NextEdge.ZMin = NextEdgeInList->ZMin;
            NextEdge.OneOverZMin = NextEdgeInList->OneOverZMin;
            NextEdge.Gradient = NextEdgeInList->Gradient;
            NextEdge.ZGradient = NextEdgeInList->ZGradient;
            NextEdge.OneOverZGradient = NextEdgeInList->OneOverZGradient;
            NextEdge.YMin = NextEdgeInList->YMin;
            NextEdge.UMin = NextEdgeInList->UMin;
            NextEdge.VMin = NextEdgeInList->VMin;
            NextEdge.UGradient = NextEdgeInList->UGradient;
            NextEdge.VGradient = NextEdgeInList->VGradient;
            NextEdge.Left = NextEdgeInList->Left;
            NextEdge.MinColor = NextEdgeInList->MinColor;
            NextEdge.ColorGradient = NextEdgeInList->ColorGradient;
            NextEdge.MinNormal = NextEdgeInList->MinNormal;
            NextEdge.NormalGradient = NextEdgeInList->NormalGradient;
            NextEdge.Next = 0;
            
            Work->Commands = Commands;
            Work->OutputTarget = Buffer;
            Work->Bitmap = Bitmap;
            Work->CurrentEdgeInList = CurrentEdge;
            Work->NextEdgeInList = NextEdge;
            Work->RowIndex = RowIndex;
            Work->PhongShading = PhongShading;
            
            Platform.AddEntry(RenderQueue, DoLineRenderWork, Work);
                        
            CurrentEdgeInList->XMin += CurrentEdgeInList->Gradient;
            NextEdgeInList->XMin += NextEdgeInList->Gradient;
            
            CurrentEdgeInList->ZMin += CurrentEdgeInList->ZGradient;
            NextEdgeInList->ZMin += NextEdgeInList->ZGradient;
            
            CurrentEdgeInList->MinColor += CurrentEdgeInList->ColorGradient;
            NextEdgeInList->MinColor += NextEdgeInList->ColorGradient;
            
            CurrentEdgeInList->MinNormal = Normalize(CurrentEdgeInList->MinNormal + CurrentEdgeInList->NormalGradient);
            NextEdgeInList->MinNormal = Normalize(NextEdgeInList->MinNormal + NextEdgeInList->NormalGradient);
            
            CurrentEdgeInList->UMin += CurrentEdgeInList->UGradient;
            CurrentEdgeInList->VMin += CurrentEdgeInList->VGradient;
            CurrentEdgeInList->OneOverZMin += CurrentEdgeInList->OneOverZGradient;
            
            NextEdgeInList->UMin += NextEdgeInList->UGradient;
            NextEdgeInList->VMin += NextEdgeInList->VGradient;
            NextEdgeInList->OneOverZMin += NextEdgeInList->OneOverZGradient;
            
            if(CurrentEdgeInList->XMin > NextEdgeInList->XMin)
            {
                CurrentEdgeInList->Next = NextEdgeInList->Next;
                NextEdgeInList->Next = CurrentEdgeInList;
                if(PreviousNextEdgeInList)
                {
                    PreviousNextEdgeInList->Next = NextEdgeInList;
                }
                CurrentEdgeInList = NextEdgeInList;
                NextEdgeInList = CurrentEdgeInList->Next;
            }
            
            if(PreviousNextEdgeInList)
            {
                if(PreviousNextEdgeInList->XMin > CurrentEdgeInList->XMin)
                {
                    PreviousNextEdgeInList->Next = CurrentEdgeInList->Next;
                    CurrentEdgeInList->Next = PreviousNextEdgeInList;
                    PreviousCurrentEdgeInList->Next = CurrentEdgeInList;
                    PreviousNextEdgeInList = CurrentEdgeInList;
                    CurrentEdgeInList = PreviousNextEdgeInList->Next;
                }
            }
            
            PreviousCurrentEdgeInList = CurrentEdgeInList;
            PreviousNextEdgeInList = NextEdgeInList;
            
            if(NextEdgeInList->Next)
            {
                CurrentEdgeInList = NextEdgeInList->Next;
                NextEdgeInList = CurrentEdgeInList->Next;
            }
            else
            {
                NextEdgeInList = 0;
            }
        }
        
    }
    
}

internal PLATFORM_WORK_QUEUE_CALLBACK(DoModelRenderWork)
{
    model_render_work *Work = (model_render_work *)Data;
    
    DrawModelOptimized(Work->OutputTarget, Work->EdgeMemory, Work->EdgeCount, Work->Commands, Work->Bitmap, Work->PhongShading);
}



internal u32
FillEdgeTable(render_entry_3d_object *Object, game_render_commands *Commands, b32 PhongShading = 0)
{
    light_data *Lights = &Commands->LightData;
    u32 TriangleCount = Object->VertexCount/3;
    
    v3 Eye = V3(0.0f, 0.0f, -1.0f);
    
    u32 VisibleEdges = 0;
    
    v4 AmbientIntensity = Lights->AmbientIntensity;
    
    for(u32 TriangleIndex = 0;
        TriangleIndex < TriangleCount;
        ++TriangleIndex)
    {
        v3 CameraVertices[] = 
        {
            ((v3 *)Object->VertexData)[3*TriangleIndex + 0] + Object->P,
            ((v3 *)Object->VertexData)[3*TriangleIndex + 1] + Object->P,
            ((v3 *)Object->VertexData)[3*TriangleIndex + 2] + Object->P,
        };
        
        v3 ProjectedVertices[] = 
        {
            ProjectVertex(CameraVertices[0], &Commands->Transform),  
            ProjectVertex(CameraVertices[1], &Commands->Transform),  
            ProjectVertex(CameraVertices[2], &Commands->Transform),  
        };
        
        v4 Colors[] = 
        {
            ((v4 *)Object->ColorData)[3*TriangleIndex + 0],
            ((v4 *)Object->ColorData)[3*TriangleIndex + 1],
            ((v4 *)Object->ColorData)[3*TriangleIndex + 2],
        };
        
        v2 UVs[] = 
        {
            ((v2 *)Object->UVData)[3*TriangleIndex + 0],
            ((v2 *)Object->UVData)[3*TriangleIndex + 1],
            ((v2 *)Object->UVData)[3*TriangleIndex + 2],
        };
        
        v3 FirstVertexNormal = Normalize(ProjectedVertices[1] - ProjectedVertices[0]);
        v3 SecondVertexNormal = Normalize(ProjectedVertices[2] - ProjectedVertices[0]);
        
        v3 Normals[] = 
        {
            ((v3 *)Object->NormalData)[3*TriangleIndex + 0],
            ((v3 *)Object->NormalData)[3*TriangleIndex + 1],
            ((v3 *)Object->NormalData)[3*TriangleIndex + 2],
        };
        
        u32 Indices[3][2] = 
        {
            {0, 1},
            {1, 2},
            {2, 0},
        };
        
        if(Inner(Eye, Cross(FirstVertexNormal, SecondVertexNormal)) > 0.0f)
        {
            
            
            for(u32 EdgeIndex = 0;
                EdgeIndex < 3;
                ++EdgeIndex)
            {
                u32 MinIndex = Indices[EdgeIndex][0];
                u32 MaxIndex = Indices[EdgeIndex][1];
                
                v3 MinVertex = ProjectedVertices[MinIndex]; 
                v3 MaxVertex = ProjectedVertices[MaxIndex]; 
                
                if(MinVertex.y > MaxVertex.y)
                {
                    v3 Temp = MinVertex;
                    MinVertex = MaxVertex;
                    MaxVertex = Temp;
                    
                    u32 TempIndex = MinIndex;
                    MinIndex = MaxIndex;
                    MaxIndex = TempIndex;
                }
                
                if(MaxVertex.y > 0)
                {
                    
                    edge_info *CurrentEdge = (edge_info *)(Object->EdgeMemory) + VisibleEdges;
                    
                    v3 FirstCameraVertex = CameraVertices[MinIndex];
                    v3 SecondCameraVertex = CameraVertices[MaxIndex];
                    
                    v3 FirstNormal = Normals[MinIndex];
                    v3 SecondNormal = Normals[MaxIndex];
                    
                    v4 FirstColor = Colors[MinIndex];
                    v4 SecondColor = Colors[MaxIndex];
                    
                    v2 FirstUV = UVs[MinIndex]; 
                    v2 SecondUV = UVs[MaxIndex]; 
                    
                    v4 MaxColor = {};
                    v3 MaxNormal = {};
                    
                    CurrentEdge->YMax = RoundR32ToS32(MaxVertex.y);
                    
                    r32 ClippedY = 0;
                    r32 t = 0.0f;
                    
                    if(MinVertex.y < 0.0f)
                    {
                        ClippedY = -MinVertex.y;
                        t = (-MinVertex.y)/(MaxVertex.y - MinVertex.y);
                    }
                    
                    CurrentEdge->YMin = (s32)Maximum(0.0f, (r32)RoundR32ToS32(MinVertex.y));
                    CurrentEdge->XMin = MinVertex.x;
                    CurrentEdge->ZMin = FirstCameraVertex.z;
                    CurrentEdge->UMin = (FirstUV.x)/MinVertex.z;
                    CurrentEdge->VMin = FirstUV.y/MinVertex.z;
                    CurrentEdge->OneOverZMin = 1.0f/MinVertex.z;
                    
                    SecondUV *= 1.0f/MaxVertex.z;
                    
                    FirstUV *= 1.0f/MinVertex.z;
                    
                    light_info *Light = Lights->Lights;
                    
                    if(PhongShading)
                    {
                        CurrentEdge->MinColor = FirstColor;
                        MaxColor = SecondColor;
                        
                        CurrentEdge->MinNormal = FirstNormal;
                        MaxNormal = SecondNormal;
                    }
                    else
                    {
                        for(u32 LightIndex = 0;
                            LightIndex < Lights->LightCount;
                            ++LightIndex, ++Light)
                        {
                            v3 LightP = Light->P;
                            v4 LightIntensity = Light->Intensity;
                            
                            v3 FirstVectorToLight = Normalize(LightP - FirstCameraVertex);
                            v3 SecondVectorToLight = Normalize(LightP - SecondCameraVertex);
                            
                            if(LightIndex == 0)
                            {
                                if(Object->Bitmap)
                                {
                                    CurrentEdge->MinColor = Hadamard(V4(1.0f, 1.0f, 1.0f, 1.0f), AmbientIntensity);
                                    MaxColor = Hadamard(V4(1.0f, 1.0f, 1.0f, 1.0f), AmbientIntensity);
                                    
                                }
                                else
                                {
                                    CurrentEdge->MinColor = Hadamard(FirstColor, AmbientIntensity);
                                    MaxColor = Hadamard(SecondColor, AmbientIntensity);
                                }
                            }
                            
                            r32 FirstDot = Clamp01(Inner(FirstVectorToLight, FirstNormal));
                            r32 SecondDot = Clamp01(Inner(SecondVectorToLight, SecondNormal));
                            
                            if(Object->Bitmap)
                            {
                                CurrentEdge->MinColor = Clamp01(CurrentEdge->MinColor + FirstDot*Hadamard(V4(1.0f, 1.0f, 1.0f, 1.0f), LightIntensity));
                                MaxColor = Clamp01(MaxColor + SecondDot*Hadamard(V4(1.0f, 1.0f, 1.0f, 1.0f), LightIntensity));
                                
                            }
                            else
                            {
                                CurrentEdge->MinColor = Clamp01(CurrentEdge->MinColor + FirstDot*Hadamard(FirstColor, LightIntensity));
                                MaxColor = Clamp01(MaxColor + SecondDot*Hadamard(SecondColor, LightIntensity));
                                
                            }
                        }
                        
                    }
                    
                    if(MinVertex.y-MaxVertex.y != 0)
                    {
                        ++VisibleEdges;
                        
                        r32 YDifference = (r32)CurrentEdge->YMax-(r32)CurrentEdge->YMin;
                        
                        CurrentEdge->ZGradient = (SecondCameraVertex.z-FirstCameraVertex.z)/(YDifference);
                        CurrentEdge->Gradient = 
                            (MaxVertex.x-MinVertex.x)/(MaxVertex.y-MinVertex.y);
                        CurrentEdge->XMin += ClippedY*CurrentEdge->Gradient;
                        CurrentEdge->ZMin += ClippedY*CurrentEdge->ZGradient;
                        
                        if(Object->Bitmap)
                        {
                            CurrentEdge->UGradient = (SecondUV.u-FirstUV.u)/(YDifference);
                            CurrentEdge->VGradient = (SecondUV.v-FirstUV.v)/(YDifference);
                            
                            CurrentEdge->UMin += ClippedY*CurrentEdge->UGradient;
                            CurrentEdge->VMin += ClippedY*CurrentEdge->VGradient;
                            
                            CurrentEdge->OneOverZGradient = (((1.0f/MaxVertex.z)-CurrentEdge->OneOverZMin)/YDifference);
                            
                            CurrentEdge->OneOverZMin += ClippedY*CurrentEdge->OneOverZGradient;
                        }
                        
                        CurrentEdge->MinColor = (1.0f-t)*CurrentEdge->MinColor + t*MaxColor;
                        
                        CurrentEdge->Left = (CurrentEdge->YMin == RoundR32ToS32(ProjectedVertices[Indices[EdgeIndex][0]].y)) ? 1 : 0;
                        CurrentEdge->Next = 0;
                        
                        CurrentEdge->ColorGradient = 
                        {
                            (MaxColor.r-CurrentEdge->MinColor.r)/(YDifference),
                            (MaxColor.g-CurrentEdge->MinColor.g)/(YDifference),
                            (MaxColor.b-CurrentEdge->MinColor.b)/(YDifference),
                            (MaxColor.a-CurrentEdge->MinColor.a)/(YDifference),
                        };
                        
                        CurrentEdge->NormalGradient = 
                        {
                            (MaxNormal.x-CurrentEdge->MinNormal.x)/(YDifference),
                            (MaxNormal.y-CurrentEdge->MinNormal.y)/(YDifference),
                            (MaxNormal.z-CurrentEdge->MinNormal.z)/(YDifference),
                        };
                        
                    }
                }
            }
        }
    }
    
    MergeSort(VisibleEdges, (edge_info *)Object->EdgeMemory, (edge_info *)Commands->SortMemory);
    
    return VisibleEdges;
    
}

internal u32 
ConstructSphere(v3 *Vertices, v4 *Colors, v3 *Normals, v2 *UVs)
{
    u32 VertexCount = 0;
    r32 Radius = 0.5f;

    u32 StepCount = 24;
    
    v4 UpColor = V4(1.0f, 0.0f, 0.0f, 1.0f);
    v4 OtherColor = V4(0.0f, 1.0f, 0.0f, 1.0f);
    v4 DownColor = V4(0.0f, 1.0f, 0.0f, 1.0f);
    
    v4 ColorIncrement = 
    {
        (DownColor.r-UpColor.r)/(r32)StepCount,
        (DownColor.g-UpColor.g)/(r32)StepCount,
        (DownColor.b-UpColor.b)/(r32)StepCount,
        (DownColor.a-UpColor.a)/(r32)StepCount,
    };
    
    r32 InclinationIncrement = Pi32/StepCount;
    r32 AzimuthIncrement = (2.0f*Pi32)/(StepCount*2);
    
    v4 CurrentColor = UpColor;
    
    for(u32 InclinationIndex = 0;
        InclinationIndex < StepCount;
        ++InclinationIndex)
    {
        for(u32 AzimuthIndex = 0;
            AzimuthIndex < StepCount*2;
            ++AzimuthIndex)
        {
            if(InclinationIndex == 0)
            {
    
                r32 Inclination = (r32)InclinationIndex*(InclinationIncrement);
                r32 NextInclination = (r32)(InclinationIndex + 1)*(InclinationIncrement);
                r32 Azimuth = (r32)AzimuthIndex*(AzimuthIncrement);
                r32 NextAzimuth = (r32)(AzimuthIndex + 1)*(AzimuthIncrement);
                
                v4 Blue = V4(0.0f, 0.0f, (1.0f + Cos(Azimuth))/2.0f, 0.0f);
                v4 NextBlue = V4(0.0f, 0.0f, (1.0f + Cos(NextAzimuth))/2.0f, 0.0f);
                
                v3 FirstVertex = V3(0.0f, 1.0f, 0.0f);
                v3 SecondVertex = V3(Sin(NextInclination)*Cos(Azimuth), Cos(NextInclination), Sin(NextInclination)*Sin(Azimuth));
                v3 ThirdVertex = V3(Sin(NextInclination)*Cos(NextAzimuth), Cos(NextInclination), Sin(NextInclination)*Sin(NextAzimuth));
                
                Vertices[VertexCount] = 
                    Radius*FirstVertex;
                Normals[VertexCount] = FirstVertex;
                UVs[VertexCount] = V2(0.5f, 0.5f);
                Colors[VertexCount++] = CurrentColor + Blue;
                
                Vertices[VertexCount] = 
                    Radius*SecondVertex;
                Normals[VertexCount] = SecondVertex;
                UVs[VertexCount] = V2(SecondVertex.x, SecondVertex.z);
                Colors[VertexCount++] = CurrentColor + ColorIncrement + Blue;

                Vertices[VertexCount] = 
                    Radius*ThirdVertex;
                Normals[VertexCount] = ThirdVertex;
                UVs[VertexCount] = V2(ThirdVertex.x, ThirdVertex.z);
                Colors[VertexCount++] = CurrentColor + ColorIncrement + NextBlue;
                
            }
            else if(InclinationIndex == (StepCount - 1))
            {
                
                r32 Inclination = (r32)InclinationIndex*(InclinationIncrement);
                r32 NextInclination = (r32)(InclinationIndex + 1)*(InclinationIncrement);
                r32 Azimuth = (r32)AzimuthIndex*(AzimuthIncrement);
                r32 NextAzimuth = (r32)(AzimuthIndex + 1)*(AzimuthIncrement);
                
                v4 Blue = V4(0.0f, 0.0f, (1.0f + Cos(Azimuth))/2.0f, 0.0f);
                v4 NextBlue = V4(0.0f, 0.0f, (1.0f + Cos(NextAzimuth))/2.0f, 0.0f);
                
                v3 FirstVertex = V3(Sin(Inclination)*Cos(Azimuth), Cos(Inclination), Sin(Inclination)*Sin(Azimuth));
                v3 SecondVertex = V3(0.0f, -1.0f, 0.0f);
                v3 ThirdVertex = V3(Sin(Inclination)*Cos(NextAzimuth), Cos(Inclination), Sin(Inclination)*Sin(NextAzimuth));
                
                Vertices[VertexCount] = 
                    Radius*FirstVertex;
                Normals[VertexCount] = FirstVertex;
                UVs[VertexCount] = V2(0.5f, 0.5f);
                Colors[VertexCount++] = CurrentColor + Blue;
                
                Vertices[VertexCount] = 
                    Radius*SecondVertex;
                Normals[VertexCount] = SecondVertex;
                UVs[VertexCount] = V2(SecondVertex.x, SecondVertex.z);
                Colors[VertexCount++] = CurrentColor + ColorIncrement + Blue;

                Vertices[VertexCount] = 
                    Radius*ThirdVertex;
                Normals[VertexCount] = ThirdVertex;
                UVs[VertexCount] = V2(ThirdVertex.x, ThirdVertex.z);
                Colors[VertexCount++] = CurrentColor + ColorIncrement + NextBlue;

            }
            else
            {
                
                r32 Inclination = (r32)InclinationIndex*(InclinationIncrement);
                r32 NextInclination = (r32)(InclinationIndex + 1)*(InclinationIncrement);
                r32 Azimuth = (r32)AzimuthIndex*(AzimuthIncrement);
                r32 NextAzimuth = (r32)(AzimuthIndex + 1)*(AzimuthIncrement);
                
                v4 Blue = V4(0.0f, 0.0f, (1.0f + Cos(Azimuth))/2.0f, 0.0f);
                v4 NextBlue = V4(0.0f, 0.0f, (1.0f + Cos(NextAzimuth))/2.0f, 0.0f);
                v3 FirstVertex = V3(Sin(Inclination)*Cos(Azimuth), Cos(Inclination), Sin(Inclination)*Sin(Azimuth));
                v3 SecondVertex = V3(Sin(NextInclination)*Cos(Azimuth), Cos(NextInclination), Sin(NextInclination)*Sin(Azimuth));
                v3 ThirdVertex = V3(Sin(NextInclination)*Cos(NextAzimuth), Cos(NextInclination), Sin(NextInclination)*Sin(NextAzimuth));
                v3 FourthVertex = V3(Sin(Inclination)*Cos(NextAzimuth), Cos(Inclination), Sin(Inclination)*Sin(NextAzimuth));
                
                Vertices[VertexCount] = 
                    Radius*FirstVertex;
                Normals[VertexCount] = FirstVertex;
                UVs[VertexCount] = V2((FirstVertex.x + 1.0f)/2.0f,
                                      (FirstVertex.y + 1.0f)/2.0f);//V2(1.0f, 1.0f);
                Colors[VertexCount++] = CurrentColor + Blue;
                
                Vertices[VertexCount] = 
                    Radius*SecondVertex;
                Normals[VertexCount] = SecondVertex;
                UVs[VertexCount] = V2((SecondVertex.x + 1.0f)/2.0f,
                                      (SecondVertex.y + 1.0f)/2.0f);//V2(1.0f, 0.0f);
                Colors[VertexCount++] = CurrentColor + ColorIncrement + Blue;

                Vertices[VertexCount] = 
                    Radius*ThirdVertex;
                Normals[VertexCount] = ThirdVertex;
                UVs[VertexCount] = V2((ThirdVertex.x + 1.0f)/2.0f,
                                      (ThirdVertex.y + 1.0f)/2.0f);//V2(0.0f, 0.0f);
                Colors[VertexCount++] = CurrentColor + ColorIncrement + NextBlue;
                
                Vertices[VertexCount] = 
                    Radius*FirstVertex;
                Normals[VertexCount] = FirstVertex;
                UVs[VertexCount] = V2((FirstVertex.x + 1.0f)/2.0f,
                                      (FirstVertex.y + 1.0f)/2.0f);//V2(1.0f, 1.0f);
                Colors[VertexCount++] = CurrentColor + Blue;
                
                Vertices[VertexCount] = 
                    Radius*ThirdVertex;
                Normals[VertexCount] = ThirdVertex;
                UVs[VertexCount] = V2((ThirdVertex.x + 1.0f)/2.0f,
                                      (ThirdVertex.y + 1.0f)/2.0f);//V2(0.0f, 0.0f);
                Colors[VertexCount++] = CurrentColor + ColorIncrement + NextBlue;
                
                Vertices[VertexCount] = 
                    Radius*FourthVertex;
                Normals[VertexCount] = FourthVertex;
                UVs[VertexCount] = V2((FourthVertex.x + 1.0f)/2.0f,
                                      (FourthVertex.y + 1.0f)/2.0f);//V2(0.0f, 1.0f);
                Colors[VertexCount++] = CurrentColor + NextBlue;
                
            }
            
        }
        
        CurrentColor = CurrentColor + ColorIncrement;
    }
    
    return VertexCount;
};
