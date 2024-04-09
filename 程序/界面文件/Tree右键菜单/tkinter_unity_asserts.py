import 界面文件.bcontext.bsfunction as bs
import 底层文件.bpformat.byfunction as by
import 底层文件

from . import tkinter_file


from . import tkinter_file

### https://github.com/HearthSim/UnityPack
flag_name_dict = { 
    1: "GameObject", 
    2: "Component", 
    3: "LevelGameManager", 
    4: "Transform", 
    5: "TimeManager", 
    6: "GlobalGameManager", 
    7: "GameManager", 
    8: "Behaviour", 
    9: "GameManager", 

    11: "AudioManager", 
    12: "ParticleAnimator", 
    13: "InputManager", 

    15: "EllipsoidParticleEmitter", 

    17: "Pipeline", 
    18: "EditorExtension", 
    19: "Physics2DSettings", 
    20: "Camera", 
    21: "Material", 
 
    23: "MeshRenderer", 

    25: "Renderer", 
    26: "ParticleRenderer", 
    27: "Texture", 
    28: "Texture2D", 
    29: "SceneSettings", 
    30: "GraphicsSettings", 
    31: "PipelineManager", 
    33: "MeshFilter", 
    35: "GameManager", 

    41: "OcclusionPortal", 
    43: "Mesh", 

    45: "Skybox", 
    46: "GameManager", 
    47: "QualitySettings", 
    48: "Shader", 
    49: "TextAsset", 
    50: "Rigidbody2D", 
    51: "Physics2DManager", 
    52: "NotificationManager", 
    53: "Collider2D", 
    54: "Rigidbody", 
    55: "PhysicsManager", 
    56: "Collider", 
    57: "Joint", 
    58: "CircleCollider2D", 
    59: "HingeJoint", 
    60: "PolygonCollider2D", 
    61: "BoxCollider2D", 
    62: "PhysicsMaterial2D", 
    63: "GameManager", 
    64: "MeshCollider", 
    65: "BoxCollider", 
    66: "SpriteCollider2D", 

    68: "EdgeCollider2D", 
    71: "AnimationManager", 
    72: "ComputeShader", 
    74: "AnimationClip", 
    75: "ConstantForce", 
    76: "WorldParticleCollider", 

    78: "TagManager", 

    81: "AudioListener", 
    82: "AudioSource", 
    83: "AudioClip", 
    84: "RenderTexture", 

    87: "MeshParticleEmitter", 
    88: "ParticleEmitter", 
    89: "Cubemap", 
    90: "Avatar", 
    91: "AnimatorController", 
    92: "GUILayer", 
    93: "RuntimeAnimatorController", 
    94: "ScriptMapper", 
    95: "Animator", 
    96: "TrailRenderer", 
    98: "DelayedCallManager", 
    102: "TextMesh", 

    104: "RenderSettings", 
    108: "Light", 
    109: "CGProgram", 
    110: "BaseAnimationTrack", 
    111: "Animation", 
    114: "MonoBehaviour", 
    115: "MonoScript", 
    116: "MonoManager", 
    117: "Texture3D", 
    118: "NewAnimationTrack", 
    119: "Projector", 
    120: "LineRenderer", 
    121: "Flare", 
    122: "Halo", 
    123: "LensFlare", 
    124: "FlareLayer", 
    125: "HaloLayer", 
    126: "NavMeshAreas", # 126: "NavMeshProjectSettings",
    127: "HaloManager", 
    128: "Font", 
    129: "PlayerSettings", 
    130: "NamedObject", 
    131: "GUITexture", 
    132: "GUIText", 
    133: "GUIElement", 
    134: "PhysicMaterial", 
    135: "SphereCollider", 
    136: "CapsuleCollider", 
    137: "SkinnedMeshRenderer", 
    138: "FixedJoint", 

    140: "RaycastCollider", 
    141: "BuildSettings", 
    142: "AssetBundle", 
    143: "CharacterController", 
    144: "CharacterJoint", 
    145: "SpringJoint", 
    146: "WheelCollider", 
    147: "ResourceManager", 
    148: "NetworkView", 
    149: "NetworkManager", 
    150: "PreloadData", 

    152: "MovieTexture", 
    153: "ConfigurableJoint", 
    154: "TerrainCollider", 
    155: "MasterServerInterface", 
    156: "TerrainData", 
    157: "LightmapSettings", 
    158: "WebCamTexture", 
    159: "EditorSettings", 
    160: "InteractiveCloth", 
    161: "ClothRenderer", 
    162: "EditorUserSettings", 
    163: "SkinnedCloth", 
    164: "AudioReverbFilter", 
    165: "AudioHighPassFilter", 
    166: "AudioChorusFilter", 
    167: "AudioReverbZone", 
    168: "AudioEchoFilter", 
    169: "AudioLowPassFilter", 
    170: "AudioDistortionFilter", 
    171: "SparseTexture", 

    180: "AudioBehaviour", 
    181: "AudioFilter", 
    182: "WindZone", 
    183: "Cloth", 
    184: "SubstanceArchive", 
    185: "ProceduralMaterial", 
    186: "ProceduralTexture", 

    191: "OffMeshLink", 
    192: "OcclusionArea", 
    193: "Tree", 
    194: "NavMeshObsolete", 
    195: "NavMeshAgent", 
    196: "NavMeshSettings", 
    197: "LightProbesLegacy", 
    198: "ParticleSystem", 
    199: "ParticleSystemRenderer", 
    200: "ShaderVariantCollection", 
    205: "LODGroup", 
    206: "BlendTree", 
    207: "Motion", 
    208: "NavMeshObstacle", 
    210: "TerrainInstance", 
    212: "SpriteRenderer", 
    213: "Sprite", 
    214: "CachedSpriteAtlas", 
    215: "ReflectionProbe", 
    216: "ReflectionProbes", 

    220: "LightProbeGroup", 
    221: "AnimatorOverrideController", 
    222: "CanvasRenderer", 
    223: "Canvas", 
    224: "RectTransform", 
    225: "CanvasGroup", 
    226: "BillboardAsset", 
    227: "BillboardRenderer", 
    228: "SpeedTreeWindAsset", 
    229: "AnchoredJoint2D", 
    230: "Joint2D", 
    231: "SpringJoint2D", 
    232: "DistanceJoint2D", 
    233: "HingeJoint2D", 
    234: "SliderJoint2D", 
    235: "WheelJoint2D", 
    238: "NavMeshData", 
    240: "AudioMixer", 
    241: "AudioMixerController", 
    243: "AudioMixerGroupController", 
    244: "AudioMixerEffectController", 
    245: "AudioMixerSnapshotController", 
    246: "PhysicsUpdateBehaviour2D", 
    247: "ConstantForce2D", 
    248: "Effector2D", 
    249: "AreaEffector2D", 
    250: "PointEffector2D", 
    251: "PlatformEffector2D", 
    252: "SurfaceEffector2D", 
    258: "LightProbes", 

    271: "SampleClip", 
    272: "AudioMixerSnapshot", 
    273: "AudioMixerGroup", 

    290: "AssetBundleManifest", 
    300: "RuntimeInitializeOnLoadManager", 

    1001: "Prefab", 
    1002: "EditorExtensionImpl", 
    1003: "AssetImporter", 
    1004: "AssetDatabase", 
    1005: "Mesh3DSImporter", 
    1006: "TextureImporter", 
    1007: "ShaderImporter", 
    1008: "ComputeShaderImporter", 

    1011: "AvatarMask", 

    1020: "AudioImporter",
    
    1026: "HierarchyState", 
    1027: "GUIDSerializer", 
    1028: "AssetMetaData", 
    1029: "DefaultAsset", 
    1030: "DefaultImporter", 
    1031: "TextScriptImporter", 
    1032: "SceneAsset", 
    1034: "NativeFormatImporter", 
    1035: "MonoImporter", 
    1037: "AssetServerCache", 
    1038: "LibraryAssetImporter", 
    1040: "ModelImporter", 
    1041: "FBXImporter", 
    1042: "TrueTypeFontImporter",
    1044: "MovieImporter", 
    1045: "EditorBuildSettings", 
    1046: "DDSImporter",
    1048: "InspectorExpandedState", 
    1049: "AnnotationManager", 
    1050: "PluginImporter",     
    1051: "EditorUserBuildSettings", 
    1052: "PVRImporter", 
    1053: "ASTCImporter", 
    1054: "KTXImporter", 

    1101: "AnimatorStateTransition", 
    1102: "AnimatorState", 
    1105: "HumanTemplate", 
    1107: "AnimatorStateMachine",
    1108: "PreviewAssetType", 
    1109: "AnimatorTransition", 
    1110: "SpeedTreeImporter", 
    1111: "AnimatorTransitionBase", 
    1112: "SubstanceImporter", 
    1113: "LightmapParameters", 
    1120: "LightmapSnapshot", 
}


class 类(tkinter_file.类):
    def __init__(self, frametreeview):
        self.frametreeview = frametreeview
        self.asserts_文件分块_wrapper = self.wrappercontext(self.asserts_文件分块)


    def asserts_文件分块(self):
        bs.insertvalue(text=f"是小端")
        by.endian("<")

        self.wrapperinsert(function=asserts_stream_head)(       self, label="stream_head")
        typeflags, offsets = [], []
        self.wrapperinsert(function=asserts_stream_type_list)(  self, label="stream_type", typeflags=typeflags)
        self.wrapperinsert(function=asserts_stream_offset_list)(self, label="stream_offset", offsets=offsets)
        self.wrapperinsert(function=asserts_stream_unknow_list)(self, label="stream_unknow")
        self.wrapperinsert(function=asserts_stream_name_list)(  self, label="stream_name")
        self.wrapperinsert(function=asserts_stream_pad0)(       self, label="stream_pad0")

        typenames = []
        for typeflag in typeflags:
            try:
                typenames.append(flag_name_dict[typeflag])
            except:
                raise ValueError(f"{[typeflag]}, 有1个或多个未识别的typeflag！")
        with bs.insertvalue(text=f"typenames"): 
            for name in typenames: bs.insertvalue(text=name)

        self.wrapperinsert(function=asserts_stream_file_list)(  self, label="stream_file", offsets=offsets, typenames=typenames)

        bs.insertblock(text=f"余下_{by.remainsize()}", bp=by.readremainslice().bp)


def asserts_stream_head(*args, **kwargs):
    bs.insertblock(text=f"_guid_20", bp=by.readslice(20).bp)
    sizeb, charb = by.readcharend0sizeseek0(), by.readcharend0seek0()
    bs.insertblock(text=f"_version_{sizeb}_{charb}", bp=by.readslice(sizeb).bp)
    bs.insertblock(text=f"_5", bp=by.readslice(5).bp)


def asserts_stream_type_list(*args, **kwargs):
    typeflags = kwargs.get("typeflags", [])

    numbe = by.readuint32seek0()
    bs.insertblock(text=f"_{numbe}_4", bp=by.readslice(4).bp)
    for i in range(numbe):
        flag = by.readuint32seek0()
        typeflags.append(flag)
        sizeb = 23
        if flag == 114: sizeb = 39
        bs.insertblock(text=f"_{i}_{sizeb}", bp=by.readslice(sizeb).bp)


def asserts_stream_offset_list(*args, **kwargs):
    offsets = kwargs.get("offsets", [])

    start = by.tell()
    numbe = by.readuint32()
    while True:
        if by.readuint8() != 0: break
    final = by.tell()
    sizeb = final - start
    by.seek(-sizeb)
    npad0 = sizeb - 4 - 1
    bs.insertblock(text=f"_{numbe}_4", bp=by.readslice(4).bp)
    bs.insertblock(text=f"_{npad0}", bp=by.readslice(npad0).bp)

    for i in range(numbe):
        info = by.readuint32seek0(5) # [index1, index2, offset, size, typeindex]
        offsets.append(info)
        bs.insertblock(text=f"_{i}_20_{info}", bp=by.readslice(20).bp)

    
def asserts_stream_unknow_list(*args, **kwargs):
    numbe = by.readuint32seek0()
    bs.insertblock(text=f"_{numbe}_4", bp=by.readslice(4).bp)
    for i in range(numbe):
        sizeb = 12
        bs.insertblock(text=f"_{i}_{sizeb}_", bp=by.readslice(sizeb).bp)


def asserts_stream_name_list(*args, **kwargs):
    number = by.readuint32seek0()
    bs.insertblock(text=f"_{number}_4", bp=by.readslice(4).bp)
    for i in range(number):
        start1 = by.tell()
        bufferuint8 = by.readuint8(21)
        bufferchars = by.readcharend0()
        final1 = by.tell()
        sizeb1 = final1 - start1
        by.seek(-sizeb1)
        bs.insertblock(text=f"_{i}_{sizeb1}_{bufferchars}", bp=by.readslice(sizeb1).bp)    


def asserts_stream_pad0(*args, **kwargs):
    numbe = 16
    sizeb = by.tell()
    npad0 = 4096 - sizeb if sizeb <= 4096 else numbe - (sizeb %  numbe) # == 0 时，正好npad0 = 16
    bs.insertblock(text=f"stream_npad0_{npad0}", bp=by.readslice(npad0).bp)


def asserts_stream_file_list(*args, **kwargs):
    offsets = kwargs.get("offsets", [])
    typenames = kwargs.get("typenames", [])

    前置偏移 = by.tell()
    for index1, index2, offset, size, typeindex in offsets:
        by.movetell(offset+前置偏移)
        with by.readsliceseek0(1024):
            name = by.readchar(by.readuint32())
        bs.insertblock(text=f"{[index1, index2, offset, size, offset+size]}__{typenames[typeindex]}__{name}", bp=by.readslice(size).bp)

                    

        
     
# def asserts_stream_offset_list(self, offsets=[]):
#     with bs.insertvalue(text=f"stream_offset_") as fitem:
#         start = by.tell()

#         numbe = by.readuint32seek0()
#         bs.insertblock(text=f"_{numbe}_4", bp=by.readslice(4).bp)

#         npad0 = 0
#         if  [by.readuint8seek0(1)] == [1]: npad0 = 0
#         if   by.readuint8seek0(2)  == [0, 1]: npad0 = 1
#         if   by.readuint8seek0(3)  == [0, 0, 1]: npad0 = 2
#         if   by.readuint8seek0(4)  == [0, 0, 0, 1]: npad0 = 3
#         bs.insertblock(text=f"_{npad0}", bp=by.readslice(npad0).bp)

#         for i in range(numbe):
#             info = by.readuint32seek0(5) # index, zero, offset, size, typeindex
#             offsets.append(info)
#             bs.insertblock(text=f"_{i}_20_{info}", bp=by.readslice(20).bp)
#         final = by.tell()
    
#     sizeb = final - start
#     by.seek(-sizeb)
#     self.frametreeview.itemblock(fitem, text=f"stream_offset_{sizeb}", bp=by.readslice(sizeb).bp)






