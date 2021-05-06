using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Numerics;
using System.Runtime.InteropServices;

public static class Program
{
    public static void Main()
    {
        using Context context = new Context();
        context.EnableAlgorithms();
        using Accelerator device = new CudaAccelerator(context);

        int width = 1920;
        int height = 1080;

        byte[] h_bitmapData = new byte[width * height * 3];

        using MemoryBuffer2D<Vec3> canvasData = device.Allocate<Vec3>(width, height);
        using MemoryBuffer<byte> d_bitmapData = device.Allocate<byte>(width * height * 3);

        CanvasData c = new CanvasData(canvasData, d_bitmapData, width, height);
                                    // pos              // look at         // up           
        Camera camera = new Camera(new Vec3(0, 50, -100), new Vec3(0, 0, 0), new Vec3(0, -1, 0), width, height, 40f);

        WorldData world = new WorldData(device);
        //world.loadMesh(new Vec3(10, 0, 0), "./Assets/defaultcube.obj");
        world.loadMesh(new Vec3(0, 0, 0), "./Assets/cat.obj");

        var frameBufferToBitmap = device.LoadAutoGroupedStreamKernel<Index2, CanvasData>(CanvasData.CanvasToBitmap);
        var RTMethod = device.LoadAutoGroupedStreamKernel<Index2, CanvasData, dWorldBuffer, Camera>(PerPixelRayIntersectionMethod);

        //do rasterization here

        Stopwatch timer = new Stopwatch();
        timer.Start();

        RTMethod(new Index2(width, height), c, world.getDeviceWorldBuffer(), camera);
        frameBufferToBitmap(canvasData.Extent, c);
        device.Synchronize();

        d_bitmapData.CopyTo(h_bitmapData, 0, 0, d_bitmapData.Extent);

        timer.Stop();
        Console.WriteLine("Rendered in: " + timer.Elapsed);

        //bitmap magic that ignores striding be careful with some 
        using Bitmap b = new Bitmap(width, height, width * 3, PixelFormat.Format24bppRgb, Marshal.UnsafeAddrOfPinnedArrayElement(h_bitmapData, 0));
        b.Save("out.bmp");

        Process.Start("cmd.exe", "/c out.bmp");
    }

    public static void PerPixelRayIntersectionMethod(Index2 pixel, CanvasData data, dWorldBuffer world, Camera camera)
    {
        int x = pixel.X;
        int y = pixel.Y;

        Ray ray = camera.GetRay(x, y);

        HitRecord hit = default;
        hit.t = float.MaxValue;

        for (int i = 0; i < world.meshes.Length; i++)
        {
            if (world.meshes[i].aabb.hit(ray, 0.001f, hit.t))
            {
                HitRecord meshHit = GetTriangleHit(ray, world, world.meshes[i], hit.t);
                if(meshHit.drawableID != -1)
                {
                    hit = meshHit;
                    data.setColor(pixel, new Vec3(1, 0, 1));
                }
            }
        }
    }

    private static HitRecord GetTriangleHit(Ray r, dWorldBuffer world, dGPUMesh mesh, float nearerThan)
    {
        Triangle t = new Triangle();
        float currentNearestDist = nearerThan;
        int NcurrentIndex = -1;
        int material = 0;
        float Ndet = 0;

        for (int i = 0; i < mesh.triangleCount; i++)
        {
            t = mesh.GetTriangle(i, world);
            Vec3 tuVec = t.uVector();
            Vec3 tvVec = t.vVector();
            Vec3 pVec = Vec3.cross(r.b, tvVec);
            float det = Vec3.dot(tuVec, pVec);

            if (XMath.Abs(det) < nearerThan)
            {
                float invDet = 1.0f / det;
                Vec3 tVec = r.a - t.Vert0;
                float u = Vec3.dot(tVec, pVec) * invDet;
                Vec3 qVec = Vec3.cross(tVec, tuVec);
                float v = Vec3.dot(r.b, qVec) * invDet;

                if (u > 0 && u <= 1.0f && v > 0 && u + v <= 1.0f)
                {
                    float temp = Vec3.dot(tvVec, qVec) * invDet;
                    if (temp < currentNearestDist)
                    {
                        currentNearestDist = temp;
                        NcurrentIndex = i;
                        Ndet = det;
                        material = t.MaterialID;
                    }
                }
            }
        }

        if (NcurrentIndex == -1)
        {
            return new HitRecord(float.MaxValue, new Vec3(), new Vec3(), false, -1, -1);
        }
        else
        {
            if (Ndet < 0)
            {
                return new HitRecord(currentNearestDist, r.pointAtParameter(currentNearestDist), -t.faceNormal(), true, material, NcurrentIndex);
            }
            else
            {
                return new HitRecord(currentNearestDist, r.pointAtParameter(currentNearestDist), t.faceNormal(), false, material, NcurrentIndex);
            }
        }
    }
}

public struct Camera
{
    public SpecializedValue<int> height;
    public SpecializedValue<int> width;

    public float verticalFov;
    public Vec3 origin;
    public Vec3 lookAt;
    public Vec3 up;
    public OrthoNormalBasis axis;

    public float aspectRatio;
    public float cameraPlaneDist;
    public float reciprocalHeight;
    public float reciprocalWidth;

    public Camera(Vec3 origin, Vec3 lookAt, Vec3 up, int width, int height, float verticalFov)
    {
        this.width = new SpecializedValue<int>(width);
        this.height = new SpecializedValue<int>(height);
        this.verticalFov = verticalFov;
        this.origin = origin;
        this.lookAt = lookAt;
        this.up = up;

        axis = OrthoNormalBasis.fromZY(Vec3.unitVector(lookAt - origin), up);

        aspectRatio = ((float)width / (float)height);
        cameraPlaneDist = 1.0f / XMath.Tan(verticalFov * XMath.PI / 360.0f);
        reciprocalHeight = 1.0f / height;
        reciprocalWidth = 1.0f / width;
    }

    private Ray rayFromUnit(float x, float y)
    {
        Vec3 xContrib = axis.x * -x * aspectRatio;
        Vec3 yContrib = axis.y * -y;
        Vec3 zContrib = axis.z * cameraPlaneDist;
        Vec3 direction = Vec3.unitVector(xContrib + yContrib + zContrib);

        return new Ray(origin, direction);
    }


    public Ray GetRay(float x, float y)
    {
        return rayFromUnit(2f * (x * reciprocalWidth) - 1f, 2f * (y * reciprocalHeight) - 1f);
    }
}

public struct HitRecord
{
    public float t;
    public bool inside;
    public Vec3 p;
    public Vec3 normal;
    public int materialID;
    public int drawableID;

    public HitRecord(float t, Vec3 p, Vec3 normal, bool inside, int materialID, int drawableID)
    {
        this.t = t;
        this.inside = inside;
        this.p = p;
        this.normal = normal;
        this.materialID = materialID;
        this.drawableID = drawableID;
    }

    public HitRecord(float t, Vec3 p, Vec3 normal, Vec3 rayDirection, int materialID, int drawableID)
    {
        this.t = t;
        inside = Vec3.dot(normal, rayDirection) > 0;
        this.p = p;
        this.normal = normal;
        this.materialID = materialID;
        this.drawableID = drawableID;
    }
}

public readonly struct Ray
{
    public readonly Vec3 a;
    public readonly Vec3 b;

    public Ray(Vec3 a, Vec3 b)
    {
        this.a = a;
        this.b = Vec3.unitVector(b);
    }

    public Vec3 pointAtParameter(float t)
    {
        return a + (t * b);
    }
}

public readonly struct OrthoNormalBasis
{
    public readonly Vec3 x;
    public readonly Vec3 y;
    public readonly Vec3 z;

    public OrthoNormalBasis(Vec3 x, Vec3 y, Vec3 z)
    {
        this.x = x;
        this.y = y;
        this.z = z;
    }


    public Vec3 transform(Vec3 pos)
    {
        return x * pos.x + y * pos.y + z * pos.z;
    }


    public static OrthoNormalBasis fromXY(Vec3 x, Vec3 y)
    {
        Vec3 zz = Vec3.unitVector(Vec3.cross(x, y));
        Vec3 yy = Vec3.unitVector(Vec3.cross(zz, x));
        return new OrthoNormalBasis(x, yy, zz);
    }


    public static OrthoNormalBasis fromYX(Vec3 y, Vec3 x)
    {
        Vec3 zz = Vec3.unitVector(Vec3.cross(x, y));
        Vec3 xx = Vec3.unitVector(Vec3.cross(y, zz));
        return new OrthoNormalBasis(xx, y, zz);
    }


    public static OrthoNormalBasis fromXZ(Vec3 x, Vec3 z)
    {
        Vec3 yy = Vec3.unitVector(Vec3.cross(z, x));
        Vec3 zz = Vec3.unitVector(Vec3.cross(x, yy));
        return new OrthoNormalBasis(x, yy, zz);
    }


    public static OrthoNormalBasis fromZX(Vec3 z, Vec3 x)
    {
        Vec3 yy = Vec3.unitVector(Vec3.cross(z, x));
        Vec3 xx = Vec3.unitVector(Vec3.cross(yy, z));
        return new OrthoNormalBasis(xx, yy, z);
    }


    public static OrthoNormalBasis fromYZ(Vec3 y, Vec3 z)
    {
        Vec3 xx = Vec3.unitVector(Vec3.cross(y, z));
        Vec3 zz = Vec3.unitVector(Vec3.cross(xx, y));
        return new OrthoNormalBasis(xx, y, zz);
    }


    public static OrthoNormalBasis fromZY(Vec3 z, Vec3 y)
    {
        Vec3 xx = Vec3.unitVector(Vec3.cross(y, z));
        Vec3 yy = Vec3.unitVector(Vec3.cross(z, xx));
        return new OrthoNormalBasis(xx, yy, z);
    }


    public static OrthoNormalBasis fromZ(Vec3 z)
    {
        Vec3 xx;
        if (XMath.Abs(Vec3.dot(z, new Vec3(1, 0, 0))) > 0.99999f)
        {
            xx = Vec3.unitVector(Vec3.cross(new Vec3(0, 1, 0), z));
        }
        else
        {
            xx = Vec3.unitVector(Vec3.cross(new Vec3(1, 0, 0), z));
        }
        Vec3 yy = Vec3.unitVector(Vec3.cross(z, xx));
        return new OrthoNormalBasis(xx, yy, z);
    }
}

public class WorldData
{
    public Accelerator device;
    public hWorldBuffer worldBuffer;

    public WorldData(Accelerator device)
    {
        this.device = device;

        worldBuffer = new hWorldBuffer(device);

    }

    public void loadMesh(Vec3 pos, string filename)
    {
        hGPUMesh loadedMesh = LoadMeshFromFile(pos, filename);
        if (loadedMesh.triangleCount > 0)
        {
            worldBuffer.addMaterial(MaterialData.makeDiffuse(new Vec3(0.5f, 0.5f, 0.5f)));
            worldBuffer.addGPUMesh(loadedMesh);
            Console.WriteLine("Loaded: " + filename);
        }
    }

    private hGPUMesh LoadMeshFromFile(Vec3 pos, string filename)
    {
        string[] lines = File.ReadAllLines(filename + (filename.EndsWith(".obj") ? "" : ".obj"));

        List<float> verticies = new List<float>();
        List<int> triangles = new List<int>();
        List<int> mats = new List<int>();

        int mat = 0;

        for (int i = 0; i < lines.Length; i++)
        {
            string line = lines[i];
            string[] split = line.Split(" ");

            if (line.Length > 0 && line[0] != '#' && split.Length >= 2)
            {
                switch (split[0])
                {
                    case "v":
                        {
                            if (double.TryParse(split[1], out double v0) && double.TryParse(split[2], out double v1) && double.TryParse(split[3], out double v2))
                            {
                                verticies.Add((float)v0);
                                verticies.Add((float)-v1);
                                verticies.Add((float)v2);
                            }
                            break;
                        }
                    case "f":
                        {
                            List<int> indexes = new List<int>();
                            for (int j = 1; j < split.Length; j++)
                            {
                                string[] indicies = split[j].Split("/");

                                if (indicies.Length >= 1)
                                {
                                    if (int.TryParse(indicies[0], out int i0))
                                    {
                                        indexes.Add(i0 < 0 ? i0 + verticies.Count : i0 - 1);
                                    }
                                }
                            }

                            for (int j = 1; j < indexes.Count - 1; ++j)
                            {
                                triangles.Add(indexes[0]);
                                triangles.Add(indexes[j]);
                                triangles.Add(indexes[j + 1]);
                                mats.Add(mat);
                            }

                            break;
                        }
                    case "usemtl":
                        {
                            // material handling happens here!
                            break;
                        }
                }

            }
        }

        return new hGPUMesh(pos, verticies, triangles, mats);
    }

    public dWorldBuffer getDeviceWorldBuffer()
    {
        return worldBuffer.GetDWorldBuffer();
    }
}

public class hWorldBuffer
{
    public Accelerator device;

    public List<MaterialData> materials;
    public List<hGPUMesh> meshes;
    public List<dGPUMesh> dmeshes;

    public List<float> verticies;
    public List<int> triangles;
    public List<int> triangleMaterials;

    private bool materialsDirty = true;
    private bool meshesDirty = true;

    public MemoryBuffer<dGPUMesh> d_dmeshes;
    public MemoryBuffer<MaterialData> d_materials;
    public MemoryBuffer<float> d_verticies;
    public MemoryBuffer<int> d_triangles;
    public MemoryBuffer<int> d_triangleMaterials;

    public hWorldBuffer(Accelerator device)
    {
        this.device = device;
        meshes = new List<hGPUMesh>();
        dmeshes = new List<dGPUMesh>();
        verticies = new List<float>();
        triangles = new List<int>();
        triangleMaterials = new List<int>();
        materials = new List<MaterialData>();
    }

    public int addMaterial(MaterialData toAdd)
    {
        if (materials.Contains(toAdd))
        {
            return materials.IndexOf(toAdd);
        }
        else
        {
            materials.Add(toAdd);
            return materials.Count - 1;
        }
    }


    public int addGPUMesh(hGPUMesh toAdd)
    {
        if (meshes.Contains(toAdd))
        {
            return meshes.IndexOf(toAdd);
        }
        else
        {
            meshes.Add(toAdd);
            updateMeshData(toAdd);
            return meshes.Count - 1;
        }
    }

    public dWorldBuffer GetDWorldBuffer()
    {
        return new dWorldBuffer(getDeviceMaterials(), getDeviceMeshes(), getDeviceVerts(), getDeviceTriangles(), getDeviceTriangleMats());
    }

    private void updateMeshData(hGPUMesh toAdd)
    {
        int vertIndex = verticies.Count;
        verticies.AddRange(toAdd.verticies);

        int triangleIndex = triangles.Count;
        triangles.AddRange(toAdd.triangles);

        int matIndex = triangleMaterials.Count;
        triangleMaterials.AddRange(toAdd.triangleMaterials);

        dmeshes.Add(new dGPUMesh(toAdd.aabb, toAdd.position, vertIndex, triangleIndex, matIndex, toAdd.triangleCount));
    }

    private ArrayView<MaterialData> getDeviceMaterials()
    {
        if (materialsDirty && materials.Count > 0)
        {
            if (d_materials != null)
            {
                d_materials.Dispose();
                d_materials = null;
            }

            var temp = materials.ToArray();
            d_materials = device.Allocate<MaterialData>(temp.Length);
            d_materials.CopyFrom(temp, LongIndex1.Zero, LongIndex1.Zero, d_materials.Extent);
            materialsDirty = false;
        }

        return d_materials;
    }

    private ArrayView<dGPUMesh> getDeviceMeshes()
    {
        if (meshesDirty && meshes.Count > 0)
        {
            if (d_dmeshes != null)
            {
                d_dmeshes.Dispose();
                d_dmeshes = null;
            }

            var temp = dmeshes.ToArray();
            d_dmeshes = device.Allocate<dGPUMesh>(temp.Length);
            d_dmeshes.CopyFrom(temp, LongIndex1.Zero, LongIndex1.Zero, d_dmeshes.Extent);

            var verts = verticies.ToArray();
            d_verticies = device.Allocate<float>(verts.Length);
            d_verticies.CopyFrom(verts, LongIndex1.Zero, LongIndex1.Zero, d_verticies.Extent);

            var triInd = triangles.ToArray();
            d_triangles = device.Allocate<int>(triInd.Length);
            d_triangles.CopyFrom(triInd, LongIndex1.Zero, LongIndex1.Zero, d_triangles.Extent);

            var triMat = triangleMaterials.ToArray();
            d_triangleMaterials = device.Allocate<int>(triMat.Length);
            d_triangleMaterials.CopyFrom(triMat, LongIndex1.Zero, LongIndex1.Zero, d_triangleMaterials.Extent);

            meshesDirty = false;
        }

        return d_dmeshes;
    }

    private ArrayView<float> getDeviceVerts()
    {
        if (meshesDirty && meshes.Count > 0)
        {
            //forceUpdate
            getDeviceMeshes();
        }

        return d_verticies;
    }

    private ArrayView<int> getDeviceTriangles()
    {
        if (meshesDirty && meshes.Count > 0)
        {
            //forceUpdate
            getDeviceMeshes();
        }

        return d_triangles;
    }

    private ArrayView<int> getDeviceTriangleMats()
    {
        if (meshesDirty && meshes.Count > 0)
        {
            //forceUpdate
            getDeviceMeshes();
        }

        return d_triangleMaterials;
    }
}

public struct dWorldBuffer
{
    public ArrayView<MaterialData> materials;

    //MeshData
    public ArrayView<dGPUMesh> meshes;
    public ArrayView<float> verticies;
    public ArrayView<int> triangles;
    public ArrayView<int> triangleMaterials;

    public dWorldBuffer( ArrayView<MaterialData> materials, ArrayView<dGPUMesh> meshes, ArrayView<float> verticies, ArrayView<int> triangles, ArrayView<int> triangleMaterials)
    {
        this.materials = materials;
        this.meshes = meshes;
        this.verticies = verticies;
        this.triangles = triangles;
        this.triangleMaterials = triangleMaterials;
    }

    public Triangle GetTriangle(int index)
    {
        // The following code sucks
        Vec3 position = new Vec3();

        for(int i = 0; i < meshes.Length; i++)
        {
            if(index >= meshes[i].trianglesStartIndex)
            {
                position = meshes[i].position;
            }
        }
        // End of sucky code (I hope LOL)

        int triangleIndex = index * 3;
        int vertexStartIndex0 = triangles[triangleIndex] * 3;
        int vertexStartIndex1 = triangles[triangleIndex + 1] * 3;
        int vertexStartIndex2 = triangles[triangleIndex + 2] * 3;

        Vec3 Vert0 = new Vec3(verticies[vertexStartIndex0], verticies[vertexStartIndex0 + 1], verticies[vertexStartIndex0 + 2]) + position;
        Vec3 Vert1 = new Vec3(verticies[vertexStartIndex1], verticies[vertexStartIndex1 + 1], verticies[vertexStartIndex1 + 2]) + position;
        Vec3 Vert2 = new Vec3(verticies[vertexStartIndex2], verticies[vertexStartIndex2 + 1], verticies[vertexStartIndex2 + 2]) + position;

        return new Triangle(Vert0, Vert1, Vert2, triangleMaterials[index]);
    }
}

public struct dGPUMesh
{
    public AABB aabb;
    public Vec3 position;
    public int verticiesStartIndex;
    public int trianglesStartIndex;
    public int triangleMaterialsStartIndex;
    public int triangleCount;

    public dGPUMesh(AABB aabb, Vec3 position, int verticiesStartIndex, int trianglesStartIndex, int triangleMaterialsStartIndex, int triangleCount)
    {
        this.aabb = aabb;
        this.position = position;
        this.verticiesStartIndex = verticiesStartIndex;
        this.trianglesStartIndex = trianglesStartIndex;
        this.triangleMaterialsStartIndex = triangleMaterialsStartIndex;
        this.triangleCount = triangleCount;
    }


    public Triangle GetTriangle(int index, dWorldBuffer world)
    {
        int triangleIndex = index * 3;
        int vertexStartIndex0 = world.triangles[triangleIndex] * 3;
        int vertexStartIndex1 = world.triangles[triangleIndex + 1] * 3;
        int vertexStartIndex2 = world.triangles[triangleIndex + 2] * 3;

        Vec3 Vert0 = new Vec3(world.verticies[vertexStartIndex0], world.verticies[vertexStartIndex0 + 1], world.verticies[vertexStartIndex0 + 2]) + position;
        Vec3 Vert1 = new Vec3(world.verticies[vertexStartIndex1], world.verticies[vertexStartIndex1 + 1], world.verticies[vertexStartIndex1 + 2]) + position;
        Vec3 Vert2 = new Vec3(world.verticies[vertexStartIndex2], world.verticies[vertexStartIndex2 + 1], world.verticies[vertexStartIndex2 + 2]) + position;

        return new Triangle(Vert0, Vert1, Vert2, world.triangleMaterials[index]);
    }
}
public struct hGPUMesh
{
    public AABB aabb;
    public Vec3 position;
    public List<float> verticies;
    public List<int> triangles;
    public List<int> triangleMaterials;
    public int triangleCount;

    public hGPUMesh(Vec3 position, List<float> verticies, List<int> triangles, List<int> triangleMaterials)
    {
        aabb = AABB.CreateFromVerticies(verticies, position);
        this.position = position;
        this.verticies = verticies;
        this.triangles = triangles;
        this.triangleMaterials = triangleMaterials;
        triangleCount = triangleMaterials.Count;
    }
}

public struct Triangle
{
    public Vec3 Vert0;
    public Vec3 Vert1;
    public Vec3 Vert2;
    public int MaterialID;

    public Triangle(Vec3 vert0, Vec3 vert1, Vec3 vert2, int MaterialID)
    {
        Vert0 = vert0;
        Vert1 = vert1;
        Vert2 = vert2;
        this.MaterialID = MaterialID;
    }


    public Vec3 uVector()
    {
        return Vert1 - Vert0;
    }


    public Vec3 vVector()
    {
        return Vert2 - Vert0;
    }


    public Vec3 faceNormal()
    {
        return Vec3.unitVector(Vec3.cross(Vert1 - Vert0, Vert2 - Vert0));
    }
}

public struct MaterialData
{
    public int type;
    public Vec3 color;
    public float ref_idx;
    public float reflectivity;
    public float reflectionConeAngleRadians;

    public MaterialData(Vec3 color, float ref_idx, float reflectivity, float reflectionConeAngleRadians, int type)
    {
        this.type = type;
        this.color = color;
        this.ref_idx = ref_idx;
        this.reflectivity = reflectivity;
        this.reflectionConeAngleRadians = reflectionConeAngleRadians;
    }

    public static MaterialData makeDiffuse(Vec3 diffuseColor)
    {
        return new MaterialData(diffuseColor, 0, 0, 0, 0);
    }

    public static MaterialData makeGlass(Vec3 diffuseColor, float ref_idx)
    {
        return new MaterialData(diffuseColor, ref_idx, 0, 0, 1);
    }

    public static MaterialData makeMirror(Vec3 diffuseColor, float fuzz)
    {
        return new MaterialData(diffuseColor, 0, 0, (fuzz < 1 ? fuzz : 1), 2);
    }

    public static MaterialData makeMirror(Vec3 diffuseColor)
    {
        return new MaterialData(diffuseColor, 0, 0, 0, 2);
    }

    public static MaterialData makeLight(Vec3 emmissiveColor)
    {
        return new MaterialData(emmissiveColor, 0, 0, 0, 3);
    }
}

public struct CanvasData
{
    public ArrayView2D<Vec3> canvas;
    public ArrayView<byte> bitmapData;
    public int width;
    public int height;

    public CanvasData(ArrayView2D<Vec3> canvas, ArrayView<byte> bitmapData, int width, int height)
    {
        this.canvas = canvas;
        this.bitmapData = bitmapData;
        this.width = width;
        this.height = height;
    }

    public void setColor(Index2 index, Vec3 c)
    {
        if ((index.X >= 0) && (index.X < canvas.Width) && (index.Y >= 0) && (index.Y < canvas.Height))
        {
            canvas[index] = c;
        }
    }

    public static void CanvasToBitmap(Index2 index, CanvasData c)
    {
        Vec3 color = c.canvas[index];

        int bitmapIndex = ((index.Y * c.width) + index.X) * 3;

        c.bitmapData[bitmapIndex] = (byte)(255.99f * color.x);
        c.bitmapData[bitmapIndex + 1] = (byte)(255.99f * color.y);
        c.bitmapData[bitmapIndex + 2] = (byte)(255.99f * color.z);

        c.canvas[index] = new Vec3(0, 0, 0);
    }
}

public struct Vec3
{
    public float x;
    public float y;
    public float z;


    public Vec3(float x, float y, float z)
    {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    public Vec3(double x, double y, double z)
    {
        this.x = (float)x;
        this.y = (float)y;
        this.z = (float)z;
    }

    public override string ToString()
    {
        return "{" + string.Format("{0:0.00}", x) + ", " + string.Format("{0:0.00}", y) + ", " + string.Format("{0:0.00}", z) + "}";
    }


    public static Vec3 operator -(Vec3 vec)
    {
        return new Vec3(-vec.x, -vec.y, -vec.z);
    }


    public float length()
    {
        return XMath.Sqrt(x * x + y * y + z * z);
    }


    public float lengthSquared()
    {
        return x * x + y * y + z * z;
    }

    public float getAt(int a)
    {
        switch (a)
        {
            case 0:
                return x;
            case 1:
                return y;
            case 2:
                return z;
            default:
                return 0;
        }
    }

    public static Vec3 setX(Vec3 v, float x)
    {
        return new Vec3(x, v.y, v.z);
    }

    public static Vec3 setY(Vec3 v, float y)
    {
        return new Vec3(v.x, y, v.z);
    }

    public static Vec3 setZ(Vec3 v, float z)
    {
        return new Vec3(v.x, v.y, z);
    }


    public static float dist(Vec3 v1, Vec3 v2)
    {
        float dx = v1.x - v2.x;
        float dy = v1.y - v2.y;
        float dz = v1.z - v2.z;
        return XMath.Sqrt(dx * dx + dy * dy + dz * dz);
    }


    public static Vec3 operator +(Vec3 v1, Vec3 v2)
    {
        return new Vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
    }


    public static Vec3 operator -(Vec3 v1, Vec3 v2)
    {
        return new Vec3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
    }


    public static Vec3 operator *(Vec3 v1, Vec3 v2)
    {
        return new Vec3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
    }


    public static Vec3 operator /(Vec3 v1, Vec3 v2)
    {
        return new Vec3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
    }


    public static Vec3 operator /(float v, Vec3 v1)
    {
        return new Vec3(v / v1.x, v / v1.y, v / v1.z);
    }


    public static Vec3 operator *(Vec3 v1, float v)
    {
        return new Vec3(v1.x * v, v1.y * v, v1.z * v);
    }


    public static Vec3 operator *(float v, Vec3 v1)
    {
        return new Vec3(v1.x * v, v1.y * v, v1.z * v);
    }


    public static Vec3 operator +(Vec3 v1, float v)
    {
        return new Vec3(v1.x + v, v1.y + v, v1.z + v);
    }


    public static Vec3 operator +(float v, Vec3 v1)
    {
        return new Vec3(v1.x + v, v1.y + v, v1.z + v);
    }


    public static Vec3 operator /(Vec3 v1, float v)
    {
        return v1 * (1.0f / v);
    }


    public static float dot(Vec3 v1, Vec3 v2)
    {
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    }


    public static Vec3 cross(Vec3 v1, Vec3 v2)
    {
        return new Vec3(v1.y * v2.z - v1.z * v2.y,
                      -(v1.x * v2.z - v1.z * v2.x),
                        v1.x * v2.y - v1.y * v2.x);
    }


    public static Vec3 unitVector(Vec3 v)
    {
        return v / XMath.Sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    }


    public static Vec3 reflect(Vec3 normal, Vec3 incomming)
    {
        return unitVector(incomming - normal * 2f * dot(incomming, normal));
    }


    public static Vec3 refract(Vec3 v, Vec3 n, float niOverNt)
    {
        Vec3 uv = unitVector(v);
        float dt = dot(uv, n);
        float discriminant = 1.0f - niOverNt * niOverNt * (1f - dt * dt);

        if (discriminant > 0)
        {
            return niOverNt * (uv - (n * dt)) - n * XMath.Sqrt(discriminant);
        }

        return v;
    }


    public static float NormalReflectance(Vec3 normal, Vec3 incomming, float iorFrom, float iorTo)
    {
        float iorRatio = iorFrom / iorTo;
        float cosThetaI = -dot(normal, incomming);
        float sinThetaTSquared = iorRatio * iorRatio * (1 - cosThetaI * cosThetaI);
        if (sinThetaTSquared > 1)
        {
            return 1f;
        }

        float cosThetaT = XMath.Sqrt(1 - sinThetaTSquared);
        float rPerpendicular = (iorFrom * cosThetaI - iorTo * cosThetaT) / (iorFrom * cosThetaI + iorTo * cosThetaT);
        float rParallel = (iorFrom * cosThetaI - iorTo * cosThetaT) / (iorFrom * cosThetaI + iorTo * cosThetaT);
        return (rPerpendicular * rPerpendicular + rParallel * rParallel) / 2f;
    }


    public static Vec3 aces_approx(Vec3 v)
    {
        v *= 0.6f;
        float a = 2.51f;
        float b = 0.03f;
        float c = 2.43f;
        float d = 0.59f;
        float e = 0.14f;
        Vec3 working = (v * (a * v + b)) / (v * (c * v + d) + e);
        return new Vec3(XMath.Clamp(working.x, 0, 1), XMath.Clamp(working.y, 0, 1), XMath.Clamp(working.z, 0, 1));
    }


    public static Vec3 reinhard(Vec3 v)
    {
        return v / (1.0f + v);
    }


    public static bool Equals(Vec3 a, Vec3 b)
    {
        return a.x == b.x &&
               a.y == b.y &&
               a.z == b.z;
    }

    public static implicit operator Vector3(Vec3 d)
    {
        return new Vector3((float)d.x, (float)d.y, (float)d.z);
    }

    public static implicit operator Vec3(Vector3 d)
    {
        return new Vec3(d.X, d.Y, d.Z);
    }

    public static implicit operator Vector4(Vec3 d)
    {
        return new Vector4((float)d.x, (float)d.y, (float)d.z, 0);
    }

    public static implicit operator Vec3(Vector4 d)
    {
        return new Vec3(d.X, d.Y, d.Z);
    }
}

public struct AABB
{
    public Vec3 min;
    public Vec3 max;

    public AABB(Vec3 min, Vec3 max)
    {
        this.min = min;
        this.max = max;
    }

    public static AABB CreateFromTriangle(Vec3 Vert0, Vec3 Vert1, Vec3 Vert2)
    {
        float minX = Vert0.x;
        float maxX = Vert0.x;

        float minY = Vert0.y;
        float maxY = Vert0.y;

        float minZ = Vert0.z;
        float maxZ = Vert0.z;

        if (Vert1.x < minX)
        {
            minX = Vert1.x;
        }

        if (Vert2.x < minX)
        {
            minX = Vert2.x;
        }

        if (Vert1.y < minY)
        {
            minY = Vert1.y;
        }

        if (Vert2.y < minY)
        {
            minY = Vert2.y;
        }

        if (Vert1.z < minZ)
        {
            minZ = Vert1.z;
        }

        if (Vert2.z < minZ)
        {
            minZ = Vert2.z;
        }

        if (Vert1.x > maxX)
        {
            maxX = Vert1.x;
        }

        if (Vert2.x > maxX)
        {
            maxX = Vert2.x;
        }

        if (Vert1.y > maxY)
        {
            maxY = Vert1.y;
        }

        if (Vert2.y > maxY)
        {
            maxY = Vert2.y;
        }

        if (Vert1.z > maxZ)
        {
            maxZ = Vert1.z;
        }

        if (Vert2.z > maxZ)
        {
            maxZ = Vert2.z;
        }

        return new AABB(new Vec3(minX, minY, minZ), new Vec3(maxX, maxY, maxZ));
    }

    public static AABB CreateFromVerticies(List<float> packedVerts, Vec3 offset)
    {
        if (packedVerts.Count < 3)
        {
            return new AABB();
        }

        float minX = packedVerts[0];
        float maxX = packedVerts[0];

        float minY = packedVerts[1];
        float maxY = packedVerts[1];

        float minZ = packedVerts[2];
        float maxZ = packedVerts[2];

        for (int i = 1; i < packedVerts.Count / 3; i++)
        {
            float x = packedVerts[i * 3];
            float y = packedVerts[i * 3 + 1];
            float z = packedVerts[i * 3 + 2];

            if (x < minX)
            {
                minX = x;
            }

            if (x > maxX)
            {
                maxX = x;
            }

            if (y < minY)
            {
                minY = y;
            }

            if (y > maxY)
            {
                maxY = y;
            }

            if (z < minZ)
            {
                minZ = z;
            }

            if (z > maxZ)
            {
                maxZ = z;
            }
        }

        return new AABB(new Vec3(minX, minY, minZ) + offset, new Vec3(maxX, maxY, maxZ) + offset);
    }

    public static AABB surrounding_box(AABB box0, AABB box1)
    {
        Vec3 small = new Vec3(XMath.Min(box0.min.x, box1.min.x), XMath.Min(box0.min.y, box1.min.y), XMath.Min(box0.min.z, box1.min.z));
        Vec3 big = new Vec3(XMath.Max(box0.max.x, box1.max.x), XMath.Max(box0.max.y, box1.max.y), XMath.Max(box0.max.z, box1.max.z));
        return new AABB(small, big);
    }


    public bool hit(Ray ray, float tMin, float tMax)
    {
        float minV = (min.x - ray.a.x) / ray.b.x;
        float maxV = (max.x - ray.a.x) / ray.b.x;
        float t1 = XMath.Max(minV, maxV);
        float t0 = XMath.Min(minV, maxV);
        tMin = XMath.Max(t0, tMin);
        tMax = XMath.Min(t1, tMax);

        if (tMax <= tMin)
        {
            return false;
        }

        minV = (min.y - ray.a.y) / ray.b.y;
        maxV = (max.y - ray.a.y) / ray.b.y;
        t1 = XMath.Max(minV, maxV);
        t0 = XMath.Min(minV, maxV);
        tMin = XMath.Max(t0, tMin);
        tMax = XMath.Min(t1, tMax);

        if (tMax <= tMin)
        {
            return false;
        }

        minV = (min.z - ray.a.z) / ray.b.z;
        maxV = (max.z - ray.a.z) / ray.b.z;
        t1 = XMath.Max(minV, maxV);
        t0 = XMath.Min(minV, maxV);
        tMin = XMath.Max(t0, tMin);
        tMax = XMath.Min(t1, tMax);

        if (tMax <= tMin)
        {
            return false;
        }

        return true;
    }
}