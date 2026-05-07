Shader "HPBR/Surface"
{
    Properties
    {
        [MainTexture] _BaseMap          ("Albedo",                    2D)          = "white" {}
        [MainColor]   _BaseColor        ("Color",                     Color)       = (1,1,1,1)
                      _BumpMap          ("Normal Map",                2D)          = "bump"  {}
                      _BumpScale        ("Normal Scale",              Float)       = 1.0
                      _MetallicGlossMap ("Metallic (R)  Smooth (A)", 2D)          = "black" {}
                      _Metallic         ("Metallic",      Range(0,1))             = 0.0
                      _Smoothness       ("Smoothness",    Range(0,1))             = 0.5
                      _OcclusionMap     ("Occlusion",                 2D)          = "white" {}
                      _OcclusionStrength("Occlusion Strength",Range(0,1))         = 1.0
                      _HeightMap        ("Height Map",                2D)          = "grey"  {}
        [PowerSlider(3.0)]
                      _HeightScale      ("Displacement",  Range(0,0.08))          = 0.0
                      _HeightSteps      ("Quality (steps)", Range(8,64))          = 24
    }

    SubShader
    {
        Tags
        {
            "RenderType"     = "Opaque"
            "RenderPipeline" = "UniversalPipeline"
            "Queue"          = "Geometry"
        }

        HLSLINCLUDE
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

        CBUFFER_START(UnityPerMaterial)
            float4 _BaseMap_ST;
            float4 _BaseColor;
            float  _BumpScale;
            float  _Metallic;
            float  _Smoothness;
            float  _OcclusionStrength;
            float  _HeightScale;
            float  _HeightSteps;
        CBUFFER_END
        ENDHLSL

        // ── Forward Lit ───────────────────────────────────────────────────────────
        Pass
        {
            Name "ForwardLit"
            Tags { "LightMode" = "UniversalForward" }

            HLSLPROGRAM
            #pragma vertex   Vert
            #pragma fragment Frag
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE
            #pragma multi_compile _ _ADDITIONAL_LIGHTS
            #pragma multi_compile_fog
            #pragma multi_compile_instancing

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            TEXTURE2D(_BaseMap);          SAMPLER(sampler_BaseMap);
            TEXTURE2D(_BumpMap);          SAMPLER(sampler_BumpMap);
            TEXTURE2D(_MetallicGlossMap); SAMPLER(sampler_MetallicGlossMap);
            TEXTURE2D(_OcclusionMap);     SAMPLER(sampler_OcclusionMap);
            TEXTURE2D(_HeightMap);        SAMPLER(sampler_HeightMap);

            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS   : NORMAL;
                float4 tangentOS  : TANGENT;
                float2 uv         : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionCS  : SV_POSITION;
                float2 uv          : TEXCOORD0;
                float3 positionWS  : TEXCOORD1;
                float3 normalWS    : TEXCOORD2;
                float3 tangentWS   : TEXCOORD3;
                float3 bitangentWS : TEXCOORD4;
                float  fogCoord    : TEXCOORD5;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings Vert(Attributes IN)
            {
                UNITY_SETUP_INSTANCE_ID(IN);
                Varyings OUT;
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(OUT);

                VertexPositionInputs posI  = GetVertexPositionInputs(IN.positionOS.xyz);
                VertexNormalInputs   normI = GetVertexNormalInputs(IN.normalOS, IN.tangentOS);

                OUT.positionCS  = posI.positionCS;
                OUT.positionWS  = posI.positionWS;
                OUT.uv          = TRANSFORM_TEX(IN.uv, _BaseMap);
                OUT.normalWS    = normI.normalWS;
                OUT.tangentWS   = normI.tangentWS;
                OUT.bitangentWS = normI.bitangentWS;
                OUT.fogCoord    = ComputeFogFactor(posI.positionCS.z);
                return OUT;
            }

            // ── Parallax Offset Mapping ───────────────────────────────────────────
            // Single-sample offset parallax — far more stable than iterative POM
            // for photogrammetry height maps (which have sharp leaf/pebble edges
            // that cause visible contour-line artefacts in ray-marching POM).
            //
            // Height convention: black (0.0) = raised, white (1.0) = recessed.
            // 0.5 − h centres the effect so mid-grey means "no shift".
            // Mip 2 smooths high-frequency noise without losing large-scale depth.
            float2 ParallaxOffset(float2 uv, float3 viewTS)
            {
                float  h     = SAMPLE_TEXTURE2D_LOD(_HeightMap, sampler_HeightMap, uv, 2).r;
                float2 shift = (viewTS.xy / max(abs(viewTS.z), 0.15)) * (0.5 - h) * _HeightScale;
                return uv - shift;
            }

            half4 Frag(Varyings IN) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(IN);

                // ── TBN ───────────────────────────────────────────────────────────
                float3 nWS   = normalize(IN.normalWS);
                float3 tWS   = normalize(IN.tangentWS);
                float3 bWS   = normalize(IN.bitangentWS);
                float3x3 TBN = float3x3(tWS, bWS, nWS);

                // ── Parallax UV ───────────────────────────────────────────────────
                float3 viewWS = GetWorldSpaceNormalizeViewDir(IN.positionWS);
                float2 uv     = (_HeightScale > 0.001)
                                ? ParallaxOffset(IN.uv, mul(TBN, viewWS))
                                : IN.uv;

                // ── Sample textures ───────────────────────────────────────────────
                half4  albedo   = SAMPLE_TEXTURE2D(_BaseMap,          sampler_BaseMap,          uv) * _BaseColor;
                half4  metGloss = SAMPLE_TEXTURE2D(_MetallicGlossMap, sampler_MetallicGlossMap, uv);
                half   occ      = lerp(1.0h, SAMPLE_TEXTURE2D(_OcclusionMap, sampler_OcclusionMap, uv).g, _OcclusionStrength);
                half3  normalTS = UnpackNormalScale(SAMPLE_TEXTURE2D(_BumpMap, sampler_BumpMap, uv), _BumpScale);
                float3 normalWS = normalize(TransformTangentToWorld(normalTS, TBN));

                // ── Lighting (direct + ambient) ───────────────────────────────────
                // Using GetMainLight() directly — more robust across URP setups
                // than UniversalFragmentPBR which requires specific renderer config.
                Light  mainLight = GetMainLight();
                half   NdotL     = saturate(dot(normalWS, mainLight.direction));
                half3  diffuse   = albedo.rgb * mainLight.color * NdotL;

                // Blinn-Phong specular
                half3  halfDir   = normalize(mainLight.direction + viewWS);
                half   NdotH     = saturate(dot(normalWS, halfDir));
                float  gloss     = metGloss.a * _Smoothness;
                half   specPow   = exp2(gloss * 10.0 + 1.0);
                half3  specular  = mainLight.color * pow(NdotH, specPow) * gloss * (1.0 - metGloss.r * _Metallic);

                // Ambient (spherical harmonics from skybox / light probes)
                half3  ambient   = SampleSH(normalWS) * albedo.rgb * occ;

                half4 color = half4(diffuse + specular + ambient, albedo.a);
                color.rgb   = MixFog(color.rgb, IN.fogCoord);
                return color;
            }
            ENDHLSL
        }

        // ── Shadow Caster ─────────────────────────────────────────────────────────
        Pass
        {
            Name "ShadowCaster"
            Tags { "LightMode" = "ShadowCaster" }
            ZWrite On
            ZTest  LEqual
            ColorMask 0
            Cull Back

            HLSLPROGRAM
            #pragma vertex   ShadowVert
            #pragma fragment ShadowFrag
            #pragma multi_compile_instancing

            struct Attr { float4 posOS : POSITION; };
            struct Vary { float4 posCS : SV_POSITION; };

            Vary  ShadowVert(Attr IN) { Vary O; O.posCS = TransformObjectToHClip(IN.posOS.xyz); return O; }
            half4 ShadowFrag(Vary IN) : SV_Target { return 0; }
            ENDHLSL
        }

        // ── Depth Only ────────────────────────────────────────────────────────────
        Pass
        {
            Name "DepthOnly"
            Tags { "LightMode" = "DepthOnly" }
            ZWrite On
            ColorMask 0
            Cull Back

            HLSLPROGRAM
            #pragma vertex   DepthVert
            #pragma fragment DepthFrag
            #pragma multi_compile_instancing

            struct Attr { float4 posOS : POSITION; };
            struct Vary { float4 posCS : SV_POSITION; };

            Vary  DepthVert(Attr IN) { Vary O; O.posCS = TransformObjectToHClip(IN.posOS.xyz); return O; }
            half4 DepthFrag(Vary IN) : SV_Target { return 0; }
            ENDHLSL
        }
    }

    FallBack "Universal Render Pipeline/Lit"
}
