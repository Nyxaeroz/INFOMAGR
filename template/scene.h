#pragma once

// -----------------------------------------------------------
// scene.h
// Simple test scene for ray tracing experiments. Goals:
// - Super-fast scene intersection
// - Easy interface: scene.FindNearest / IsOccluded
// - With normals and albedo: GetNormal / GetAlbedo
// - Area light source (animated), for light transport
// - Primitives can be hit from inside - for dielectrics
// - Can be extended with other primitives and/or a BVH
// - Optionally animated - for temporal experiments
// - Not everything is axis aligned - for cache experiments
// - Can be evaluated at arbitrary time - for motion blur
// - Has some high-frequency details - for filtering
// Some speed tricks that severely affect maintainability
// are enclosed in #ifdef SPEEDTRIX / #endif. Mind these
// if you plan to alter the scene in any way.
// -----------------------------------------------------------

#define SPEEDTRIX

#define PLANE_X(o,i) {if((t=-(ray.O.x+o)*ray.rD.x)<ray.t)ray.t=t,ray.objIdx=i;}
#define PLANE_Y(o,i) {if((t=-(ray.O.y+o)*ray.rD.y)<ray.t)ray.t=t,ray.objIdx=i;}
#define PLANE_Z(o,i) {if((t=-(ray.O.z+o)*ray.rD.z)<ray.t)ray.t=t,ray.objIdx=i;}

namespace Tmpl8 {

__declspec(align(64)) class Ray
{
public:
	Ray() = default;
	Ray( float3 origin, float3 direction, float distance = 1e34f )
	{
		O = origin, D = direction, t = distance;
		// calculate reciprocal ray direction for triangles and AABBs
		rD = float3( 1 / D.x, 1 / D.y, 1 / D.z );
	#ifdef SPEEDTRIX
		d0 = d1 = d2 = 0;
	#endif
	}
	float3 IntersectionPoint() { return O + t * D; }
	// ray data
#ifndef SPEEDTRIX
	float3 O, D, rD;
#else
	union { struct { float3 O; float d0; }; __m128 O4; };
	union { struct { float3 D; float d1; }; __m128 D4; };
	union { struct { float3 rD; float d2; }; __m128 rD4; };
#endif
	float t = 1e34f;
	int objIdx = -1;
	bool inside = false; // true when in medium
};
// -----------------------------------------------------------
// Material class
// -----------------------------------------------------------
class Material
{
public:
	Material() = default;
	Material(Mat t, float3 c) : type(t), color(c) {}

	Mat type = Mat::DIFFUSE;
	float3 color = float3(100, 100, 100);

};
// -----------------------------------------------------------
// Sphere primitive
// Basic sphere, with explicit support for rays that start
// inside it. Good candidate for a dielectric material.
// -----------------------------------------------------------
class Sphere
{
public:
	Sphere() = default;
	Sphere( int idx, float3 p, float r, Material m = Material()) :
		pos( p ), r2( r* r ), invr( 1 / r ), objIdx( idx ), material(m) {}
	void Intersect( Ray& ray ) const
	{
		float3 oc = ray.O - this->pos;
		float b = dot( oc, ray.D );
		float c = dot( oc, oc ) - this->r2;
		float t, d = b * b - c;
		if (d <= 0) return;
		d = sqrtf( d ), t = -b - d;
		if (t < ray.t && t > 0)
		{
			ray.t = t, ray.objIdx = objIdx;
			return;
		}
		t = d - b;
		if (t < ray.t && t > 0)
		{
			ray.t = t, ray.objIdx = objIdx;
			return;
		}
	}
	float3 GetNormal( const float3 I ) const
	{
		return (I - this->pos) * invr;
	}
	float3 GetAlbedo( const float3 I ) const
	{
		return float3( 0.93f );
	}
	float3 pos = 0;
	float r2 = 0, invr = 0;
	Material material;
	int objIdx = -1;
};

// -----------------------------------------------------------
// Plane primitive
// Basic infinite plane, defined by a normal and a distance
// from the origin (in the direction of the normal).
// -----------------------------------------------------------
class Plane
{
public:
	Plane() = default;
	Plane( int idx, float3 normal, float dist, Material m = Material()) : N( normal ), d( dist ), objIdx( idx ), material( m ) {}
	void Intersect( Ray& ray ) const
	{
		float t = -(dot( ray.O, this->N ) + this->d) / (dot( ray.D, this->N ));
		if (t < ray.t && t > 0) ray.t = t, ray.objIdx = objIdx;
	}
	float3 GetNormal( const float3 I ) const
	{
		return N;
	}
	float3 GetAlbedo( const float3 I ) const
	{
		if (N.y == 1)
		{
			// floor albedo: checkerboard
			int ix = (int)(I.x * 2 + 96.01f);
			int iz = (int)(I.z * 2 + 96.01f);
			// add deliberate aliasing to two tile
			if (ix == 98 && iz == 98) ix = (int)(I.x * 32.01f), iz = (int)(I.z * 32.01f);
			if (ix == 94 && iz == 98) ix = (int)(I.x * 64.01f), iz = (int)(I.z * 64.01f);
			return float3( ((ix + iz) & 1) ? 1 : 0.3f );
		}
		else if (N.z == -1)
		{
			// back wall: logo
			static Surface logo( "assets/logo.png" );
			int ix = (int)((I.x + 4) * (128.0f / 8));
			int iy = (int)((2 - I.y) * (64.0f / 3));
			uint p = logo.pixels[(ix & 127) + (iy & 63) * 128];
			uint3 i3( (p >> 16) & 255, (p >> 8) & 255, p & 255 );
			return float3( i3 ) * (1.0f / 255.0f);
		}
		return float3( 0.93f );
	}
	float3 N;
	float d;
	Material material;
	int objIdx = -1;
};

// -----------------------------------------------------------
// Cube primitive
// Oriented cube. Unsure if this will also work for rays that
// start inside it; maybe not the best candidate for testing
// dielectrics.
// -----------------------------------------------------------
class Cube
{
public:
	Cube() = default;
	Cube( int idx, float3 pos, float3 size, Material m = Material(), mat4 transform = mat4::Identity())
	{
		objIdx = idx;
		b[0] = pos - 0.5f * size, b[1] = pos + 0.5f * size;
		M = transform, invM = transform.FastInvertedTransformNoScale();
		material = m;
	}
	void Intersect( Ray& ray ) const
	{
		// 'rotate' the cube by transforming the ray into object space
		// using the inverse of the cube transform.
		float3 O = TransformPosition( ray.O, invM );
		float3 D = TransformVector( ray.D, invM );
		float rDx = 1 / D.x, rDy = 1 / D.y, rDz = 1 / D.z;
		int signx = D.x < 0, signy = D.y < 0, signz = D.z < 0;
		float tmin = (b[signx].x - O.x) * rDx;
		float tmax = (b[1 - signx].x - O.x) * rDx;
		float tymin = (b[signy].y - O.y) * rDy;
		float tymax = (b[1 - signy].y - O.y) * rDy;
		if (tmin > tymax || tymin > tmax) return;
		tmin = max( tmin, tymin ), tmax = min( tmax, tymax );
		float tzmin = (b[signz].z - O.z) * rDz;
		float tzmax = (b[1 - signz].z - O.z) * rDz;
		if (tmin > tzmax || tzmin > tmax) return;
		tmin = max( tmin, tzmin ), tmax = min( tmax, tzmax );
		if (tmin > 0)
		{
			if (tmin < ray.t) ray.t = tmin, ray.objIdx = objIdx;
		}
		else if (tmax > 0)
		{
			if (tmax < ray.t) ray.t = tmax, ray.objIdx = objIdx;
		}
	}
	float3 GetNormal( const float3 I ) const
	{
		// transform intersection point to object space
		float3 objI = TransformPosition( I, invM );
		// determine normal in object space
		float3 N = float3( -1, 0, 0 );
		float d0 = fabs( objI.x - b[0].x ), d1 = fabs( objI.x - b[1].x );
		float d2 = fabs( objI.y - b[0].y ), d3 = fabs( objI.y - b[1].y );
		float d4 = fabs( objI.z - b[0].z ), d5 = fabs( objI.z - b[1].z );
		float minDist = d0;
		if (d1 < minDist) minDist = d1, N.x = 1;
		if (d2 < minDist) minDist = d2, N = float3( 0, -1, 0 );
		if (d3 < minDist) minDist = d3, N = float3( 0, 1, 0 );
		if (d4 < minDist) minDist = d4, N = float3( 0, 0, -1 );
		if (d5 < minDist) minDist = d5, N = float3( 0, 0, 1 );
		// return normal in world space
		return TransformVector( N, M );
	}
	float3 GetAlbedo( const float3 I ) const
	{
		return float3( 0.1, 0.1, 0.93 );
	}
	float3 b[2];
	mat4 M, invM;
	Material material;
	int objIdx = -1;
};

// -----------------------------------------------------------
// Quad primitive
// Oriented quad, intended to be used as a light source.
// -----------------------------------------------------------
class Quad
{
public:
	Quad() = default;
	Quad( int idx, float s, mat4 transform = mat4::Identity(), Material m = Material())
	{
		objIdx = idx;
		size = s * 0.5f;
		T = transform, invT = transform.FastInvertedTransformNoScale();
		material = m;
	}
	void Intersect( Ray& ray ) const
	{
		const float3 O = TransformPosition( ray.O, invT );
		const float3 D = TransformVector( ray.D, invT );
		const float t = O.y / -D.y;
		if (t < ray.t && t > 0)
		{
			float3 I = O + t * D;
			if (I.x > -size && I.x < size && I.z > -size && I.z < size)
				ray.t = t, ray.objIdx = objIdx;
		}
	}
	float3 GetNormal( const float3 I ) const
	{
		// TransformVector( float3( 0, -1, 0 ), T ) 
		return float3( -T.cell[1], -T.cell[5], -T.cell[9] );
	}
	float3 GetAlbedo( const float3 I ) const
	{
		return float3( 10 );
	}
	float size;
	mat4 T, invT;
	Material material;
	int objIdx = -1;
};
// -----------------------------------------------------------
// Triangle primitive
// Oriented triangle, defined by three points in space
// -----------------------------------------------------------
class Triangle
{
public:
	Triangle() = default;
	Triangle(int idx, float3 q1, float3 q2, float3 q3, Material m = Material())
	{
		objIdx = idx, p1 = q1, p2 = q2, p3 = q3, material = m;
		float3 center = (p1 + p2 + p3) / 3;
		normal = cross(p2 - p1, p3 - p1);
		dist = sqrtf(center.x*center.x+center.y*center.y+center.z*center.z);
		material = m;
	}
	void Intersect(Ray& ray) const
	{
		// intersection with support plane
		float t = -(dot(ray.O, normal) + dist) / (dot(ray.D, normal));
		if (t < ray.t && t > 0) {
			float3 q = ray.O + t * ray.D;
			// determine if intersection is inside triangle by checking if cross products point in the same direction as the normal
			bool inTriangle =  dot(cross(p2 - p1, q - p1), normal) >= 0
							&& dot(cross(p3 - p2, q - p2), normal) >= 0
							&& dot(cross(p1 - p3, q - p3), normal) >= 0;
			if (inTriangle) ray.t = t, ray.objIdx = objIdx;
		}
	}
	float3 GetNormal(const float3 I) const
	{
		// may need second option based on closest side?
		return cross(p2 - p1, p3 - p1);
	}
	float3 GetAlbedo(const float3 I) const
	{
		return float3(1,0,0);
	}
	float3 p1, p2, p3, normal;
	float dist;
	Material material;
	int objIdx = -1;
};
// -----------------------------------------------------------
// Triangle primitive
// Torus based on two radii
// (Partially taken from Jacco Bikker in Teams channel)
// -----------------------------------------------------------
class Torus
{
public:
	Torus() = default;
	Torus(int idx, float a, float b, Material m = Material())
	{
		objIdx = idx;
		A = a;
		B = b;
		rc2 = a * a, rt2 = b * b;
		T = invT = mat4::Identity();
		r2 = sqrf(a + b);
		material = m;
	}
	void Intersect(Ray& ray) const
{
	// via: https://www.shadertoy.com/view/4sBGDy
	float3 O = TransformPosition(ray.O, invT);
	float3 D = TransformVector(ray.D, invT);
	// extension rays need double precision for the quadratic solver!
	double po = 1, m = dot(O, O), k3 = dot(O, D), k32 = k3 * k3;
	// bounding sphere test
	double v = k32 - m + r2;
	if (v < 0) return;
	// setup torus intersection
	double k = (m - rt2 - rc2) * 0.5, k2 = k32 + rc2 * D.z * D.z + k;
	double k1 = k * k3 + rc2 * O.z * D.z, k0 = k * k + rc2 * O.z * O.z - rc2 * rt2;
	// solve quadratic equation
	if (fabs(k3 * (k32 - k2) + k1) < 0.0001)
	{
		swap(k1, k3);
		po = -1, k0 = 1 / k0, k1 = k1 * k0, k2 = k2 * k0, k3 = k3 * k0, k32 = k3 * k3;
	}
	double c2 = 2 * k2 - 3 * k32, c1 = k3 * (k32 - k2) + k1;
	double c0 = k3 * (k3 * (-3 * k32 + 4 * k2) - 8 * k1) + 4 * k0;
	c2 /= 3, c1 *= 2, c0 /= 3;
	double Q = c2 * c2 + c0, R = 3 * c0 * c2 - c2 * c2 * c2 - c1 * c1;
	double h = R * R - Q * Q * Q, z = 0;
	if (h < 0)
	{
		const double sQ = sqrt(Q);
		z = 2 * sQ * cos(acos(R / (sQ * Q)) * 0.33333333333);
	}
	else
	{
		const double sQ = pow(sqrt(h) + fabs(R), 0.33333333333);
		z = copysign(fabs(sQ + Q / sQ), R);
	}
	z = c2 - z;
	double d1 = z - 3 * c2, d2 = z * z - 3 * c0;
	if (fabs(d1) < 1.0e-8)
	{
		if (d2 < 0) return;
		d2 = sqrt(d2);
	}
	else
	{
		if (d1 < 0) return;
		d1 = sqrt(d1 * 0.5), d2 = c1 / d1;
	}
	double t = 1e20;
	h = d1 * d1 - z + d2;
	if (h > 0)
	{
		h = sqrt(h);
		double t1 = -d1 - h - k3, t2 = -d1 + h - k3;
		t1 = (po < 0) ? 2 / t1 : t1, t2 = (po < 0) ? 2 / t2 : t2;
		if (t1 > 0) t = t1;
		if (t2 > 0) t = min(t, t2);
	}
	h = d1 * d1 - z - d2;
	if (h > 0)
	{
		h = sqrt(h);
		double t1 = d1 - h - k3, t2 = d1 + h - k3;
		t1 = (po < 0) ? 2 / t1 : t1, t2 = (po < 0) ? 2 / t2 : t2;
		if (t1 > 0) t = min(t, t1);
		if (t2 > 0) t = min(t, t2);
	}
	float ft = (float)t;
	if (ft > 0 && ft < ray.t) ray.t = ft, ray.objIdx = objIdx;
}
	float3 GetNormal( const float3 I) const { 
		// transform intersection point to object space
		float3 objI = TransformPosition(I, invT);
		// Note: for now, torus lies on xy plane with origin (0,0,0)
		// Used explanation from http://cosinekitty.com/raytrace/chapter13_torus.html
		float3 Q = A * float3(objI.x, objI.y, 0) / sqrt(objI.x * objI.x + objI.y * objI.y);
		float3 N = normalize(objI - Q);
		// return normal in world space
		return TransformVector(N, T);;
	}
	float3 GetAlbedo( const float3 I ) const { return float3( 0.93f, 0.01f, 0.01f ); }
	float A, B, rc2, rt2, r2;
	mat4 T, invT;
	Material material;
	int objIdx = -1;

};

// -----------------------------------------------------------
// Tri primitive
// Encompassing object based on triangle architecture
// -----------------------------------------------------------
class Tri {
public:
	Tri() = default;
	Tri(int idx, int type, float3 w0, float3 w1, float3 w2, Material m = Material()) {
		assert(0 <= type && type <= 2);
		objIdx = idx;
		shapeType = type; 
		material = m;

		switch (shapeType) {
		case 0:
			v0 = w0;
			v1 = w1;
			v2 = w2;
			centroid = (v0 + v1 + v2) / 3;
			normal = cross(v1 - v0, v2 - v0);
			dist = sqrtf(centroid.x * centroid.x + centroid.y * centroid.y + centroid.z * centroid.z);
			break;
		case 1:
			pos = w0;
			r2 = abs(w1.x * w1.x);
			invr = 1 / w1.x;
			break;
		case 2:
			normal = w0;
			dist = w1.x;
			break;
		}
	}
	
	void Intersect(Ray& ray) const { 
		switch (shapeType) {
			float t;
			{	case 0:
				t = -(dot(ray.O, normal) + dist) / (dot(ray.D, normal));
				if (t < ray.t && t > 0) {
					float3 q = ray.O + t * ray.D;
					// determine if intersection is inside triangle by checking if cross products point in the same direction as the normal
					bool inTriangle = dot(cross(v1 - v0, q - v0), normal) >= 0
								   && dot(cross(v2 - v1, q - v1), normal) >= 0
								   && dot(cross(v0 - v2, q - v2), normal) >= 0;
					if (inTriangle) ray.t = t, ray.objIdx = objIdx;
				}
				break;
			}
			{	case 1:
				float3 oc = ray.O - this->pos;
				float b = dot(oc, ray.D);
				float c = dot(oc, oc) - this->r2;
				float d = b * b - c;
				if (d <= 0) return;
				d = sqrtf(d), t = -b - d;
				if (t < ray.t && t > 0)
				{
					ray.t = t, ray.objIdx = objIdx;
					return;
				}
				t = d - b;
				if (t < ray.t && t > 0)
				{
					ray.t = t, ray.objIdx = objIdx;
					return;
				}
				break;
			}
			{	case 2:
				t = -(dot(ray.O, normal) + dist) / (dot(ray.D, normal));
				if (t < ray.t && t > 0) ray.t = t, ray.objIdx = objIdx;
				break;
			}
		}
	}
	float3 GetNormal( const float3 I ) const { 
		switch (shapeType) {
		case 0:
			return cross(v1 - v0, v2 - v0);
		case 1:
			return (I - this->pos) * invr;
		case 2:
			return normal;
		}
	}
	float3 GetAlbedo(const float3 I) const { 
		switch (shapeType) {
		case 0:
			return float3( 1, 0, 0 );
		case 1:
			return float3( 0.93f );
		case 2:
			if (normal.y == 1)
			{
				// floor albedo: checkerboard
				int ix = (int)(I.x * 2 + 96.01f);
				int iz = (int)(I.z * 2 + 96.01f);
				// add deliberate aliasing to two tile
				if (ix == 98 && iz == 98) ix = (int)(I.x * 32.01f), iz = (int)(I.z * 32.01f);
				if (ix == 94 && iz == 98) ix = (int)(I.x * 64.01f), iz = (int)(I.z * 64.01f);
				return float3(((ix + iz) & 1) ? 1 : 0.3f);
			}
			return float3( 0.93f );
		}
	}

	int objIdx = -1;
	int shapeType = 0; // 0 Triangle -- 1 Sphere -- 2 Plane
	Material material;
	
	// for triangles, first vertex
	// for spheres, (unused)
	// flor planes, (unused)
	union { float3 v0; };

	// for triangles, second vertex
	// for sphere, radius squared, inversed radius and (unused)
	// for planes, (unused)
	union { float3 v1; struct { float r2, invr, u1; }; };

	// for triangles, third vertex
	// for spheres, (unused)
	// for planes, (unused)
	union { float3 v2; };
	
	// used for triangles and planes
	union { float3 normal; };

	// used for triangles and planes
	union { float dist; };

	union { float3 centroid; float3 pos; };
};

// -----------------------------------------------------------
// Scene class
// We intersect this. The query is internally forwarded to the
// list of primitives, so that the nearest hit can be returned.
// For this hit (distance, obj id), we can query the normal and
// albedo.
// -----------------------------------------------------------
class Scene
{
public:
	Scene()
	{
		Tri t = Tri(-1, 1, float3(0), float3(1), float3(2));
		Material diffuse = Material(Mat::DIFFUSE, float3(1, 1, 1));
		Material mirror = Material(Mat::MIRROR, float3(1, 1, 1));
		Material glass = Material(Mat::GLASS, float3(1, 1, 1));

		// we store all primitives in one continuous buffer
		// quad = Quad( 0, 1 );									// 0: light source
		mat4 L1base = mat4::Translate(float3(-1, 2.6f, 2));
		mat4 L1 = L1base * mat4::Translate(float3(0, -0.9, 0));
		lights[0] = Quad(12, 0.5, L1);
		mat4 L2base = mat4::Translate(float3(1, 2.6f, 2));
		mat4 L2 = L2base * mat4::Translate(float3(0, -0.9, 0));
		lights[1] = Quad(13, 0.5, L2);
		sphere = Sphere( 1, float3( 0 ), 0.5f, mirror );				// 1: bouncing ball
		sphere2 = Sphere( 2, float3( 0, 2.5f, -3.07f ), 8 );	// 2: rounded corners
		cube = Cube( 3, float3( 0 ), float3( 1.15f ), glass );			// 3: cube
		plane[0] = Plane( 4, float3( 1, 0, 0 ), 3 );			// 4: left wall
		plane[1] = Plane( 5, float3( -1, 0, 0 ), 2.99f);		// 5: right wall
		plane[2] = Plane( 6, float3( 0, 1, 0 ), 1 );			// 6: floor
		plane[3] = Plane( 7, float3( 0, -1, 0 ), 2 );			// 7: ceiling
		plane[4] = Plane( 8, float3( 0, 0, 1 ), 3 );			// 8: front wall
		plane[5] = Plane( 9, float3( 0, 0, -1 ), 3.99f );		// 9: back wall
		//triangle = Triangle(10, float3(-0.5,-1,0.5f), float3(-0.5,0,0.5f), float3(-1,-0.5f,0.75f));
		torus = Torus(11, 0.5f, 0.25f);
		SetTime( 0 );
		// Note: once we have triangle support we should get rid of the class
		// hierarchy: virtuals reduce performance somewhat.
	}
	void SetTime( float t )
	{
		// default time for the scene is simply 0. Updating/ the time per frame 
		// enables animation. Updating it per ray can be used for motion blur.
		animTime = t;
		// light source animation: swing
		// mat4 M1base = mat4::Translate( float3( 0, 2.6f, 2 ) );
		// mat4 M1 = M1base * mat4::RotateZ( sinf( animTime * 0.6f ) * 0.1f ) * mat4::Translate( float3( 0, -0.9, 0 ) );
		// quad.T = M1, quad.invT = M1.FastInvertedTransformNoScale();
		// cube animation: spin
		mat4 M2base = mat4::RotateX( PI / 4 ) * mat4::RotateZ( PI / 4 );
		mat4 M2 = mat4::Translate( float3( 1.4f, 0, 2 ) ) * mat4::RotateY( animTime * 0.5f ) * M2base;
		cube.M = M2, cube.invM = M2.FastInvertedTransformNoScale();
		// torus animation: spin
		mat4 M3base = mat4::RotateX(PI / 4) * mat4::RotateZ(PI / 4);
		mat4 M3 = mat4::Translate(float3(0, 0, 2)) * mat4::RotateY(animTime * 0.5f) * M2base;
		torus.T = M3, torus.invT = M3.FastInvertedTransformNoScale();
		// sphere animation: bounce
		float tm = 1 - sqrf( fmodf( animTime, 2.0f ) - 1 );
		sphere.pos = float3( -1.4f, -0.5f + tm, 2 );
	}
	float3 GetLightPos() const
	{
		// light point position is the middle of the swinging quad
		float3 corner1 = TransformPosition( float3( -0.5f, 0, -0.5f ), quad.T );
		float3 corner2 = TransformPosition( float3( 0.5f, 0, 0.5f ), quad.T );
		return (corner1 + corner2) * 0.5f - float3( 0, 0.01f, 0 );
	}
	Quad GetRandomLight(float& pdf)
	{
		int length = sizeof(lights) / sizeof(Quad);
		// is it correct?
		int r = (int) (RandomFloat() * length);
		int index = r % length;
		
		pdf = 1.0 / length;

		if (index > length - 1)
		{
			return lights[length - 1];
		}

		return lights[index];
	}
	float3 GetRandomPosOnLight(Quad quad, float& pdf)
	{
		pdf = 1.0 / (quad.size * 2 * quad.size * 2);

		float randX = (RandomFloat() - 0.5f) * quad.size;
		float randY = (RandomFloat() - 0.5f) * quad.size;

		return TransformPosition(float3(randX, 0, randY), quad.T) - float3(0, 0.01f, 0);
	}
	float3 GetLightColor() const
	{
		return float3( 2, 2, 2 );
	}
	void FindNearest( Ray& ray ) const
	{
		// room walls - ugly shortcut for more speed
		float t;
		if (ray.D.x < 0) PLANE_X( 3, 4 ) else PLANE_X( -2.99f, 5 );
		if (ray.D.y < 0) PLANE_Y( 1, 6 ) else PLANE_Y( -2, 7 );
		if (ray.D.z < 0) PLANE_Z( 3, 8 ) else PLANE_Z( -3.99f, 9 );
		int lights_count = sizeof(lights) / sizeof(Quad);
		for (int i = 0; i < lights_count; i++)
		{
			lights[i].Intersect( ray );
		}
		//quad.Intersect( ray );
		sphere.Intersect( ray );
		sphere2.Intersect( ray );
		//cube.Intersect( ray );
		//triangle.Intersect( ray );
		torus.Intersect( ray );
	}
	bool IsOccluded( Ray& ray ) const
	{
		float rayLength = ray.t;
		// skip planes: it is not possible for the walls to occlude anything
		int lights_count = sizeof(lights) / sizeof(Quad);
		for (int i = 0; i < lights_count; i++)
		{
			lights[i].Intersect(ray);
		}
		//quad.Intersect( ray );
		sphere.Intersect( ray );
		sphere2.Intersect( ray );
		//cube.Intersect( ray );
		//triangle.Intersect( ray );
		torus.Intersect( ray );
		return ray.t < rayLength;
		// technically this is wasteful: 
		// - we potentially search beyond rayLength
		// - we store objIdx and t when we just need a yes/no
		// - we don't 'early out' after the first occlusion
	}
	float3 GetNormal( int objIdx, float3 I, float3 wo ) const
	{
		// we get the normal after finding the nearest intersection:
		// this way we prevent calculating it multiple times.
		if (objIdx == -1) return float3( 0 ); // or perhaps we should just crash
		float3 N;
		if (objIdx == 0) N = quad.GetNormal(I);
		else if (objIdx == 1) N = sphere.GetNormal(I);
		else if (objIdx == 2) N = sphere2.GetNormal(I);
		else if (objIdx == 3) N = cube.GetNormal(I);
		else if (objIdx == 10) N = triangle.GetNormal(I);
		else if (objIdx == 11) N = torus.GetNormal(I);
		else if (objIdx == 12) N = lights[0].GetNormal(I);
		else if (objIdx == 13) N = lights[1].GetNormal(I);
		else 
		{
			// faster to handle the 6 planes without a call to GetNormal
			N = float3( 0 );
			N[(objIdx - 4) / 2] = 1 - 2 * (float)(objIdx & 1);
		}
		if (dot( N, wo ) > 0) N = -N; // hit backside / inside
		return N;
	}
	float3 GetAlbedo( int objIdx, float3 I ) const
	{
		if (objIdx == -1) return float3( 0 ); // or perhaps we should just crash
		if (objIdx == 0) return quad.GetAlbedo( I );
		if (objIdx == 1) return sphere.GetAlbedo( I );
		if (objIdx == 2) return sphere2.GetAlbedo( I );
		if (objIdx == 3) return cube.GetAlbedo( I );
		if (objIdx == 10) return triangle.GetAlbedo( I );
		if (objIdx == 11) return torus.GetAlbedo( I );
		if (objIdx == 12) return lights[0].GetAlbedo(I);
		if (objIdx == 13) return lights[1].GetAlbedo(I);
		return plane[objIdx - 4].GetAlbedo( I );
		// once we have triangle support, we should pass objIdx and the bary-
		// centric coordinates of the hit, instead of the intersection location.
	}
	float GetReflectivity( int objIdx, float3 I ) const
	{
		if (objIdx == 1 /* ball */) return 1;
		if (objIdx == 6 /* floor */) return 0.3f;
		return 0;
	}
	float GetRefractivity( int objIdx, float3 I ) const
	{
		return objIdx == 3 ? 1.0f : 0.0f;
	}
	Material GetMaterial(int objIdx) const 
	{
		if (objIdx == -1) return Material(); // or perhaps we should just crash
		if (objIdx == 0) return quad.material;
		if (objIdx == 1) return sphere.material;
		if (objIdx == 2) return sphere2.material;
		if (objIdx == 3) return cube.material;
		if (objIdx == 10) return triangle.material;
		if (objIdx == 11) return torus.material;
		if (objIdx == 12) return lights[0].material;
		if (objIdx == 13) return lights[1].material;
		return plane[objIdx - 4].material;
	}



	__declspec(align(64)) // start a new cacheline here
	float animTime = 0;
	Quad quad;
	Quad lights[2];
	Sphere sphere;
	Sphere sphere2;
	Cube cube;
	Plane plane[6];
	Triangle triangle;
	Torus torus;
};

class TriScene
{
public:
	TriScene()
	{
		Material diffuse = Material(Mat::DIFFUSE, float3(1, 1, 1));
		Material mirror = Material(Mat::MIRROR, float3(1, 1, 1));
		Material glass = Material(Mat::GLASS, float3(1, 1, 1));

		light = Tri(-1, 0, float3(-0.5,2,0), float3(0.5, 2, 0), float3(0.25, 2, 0.38));

		objects[0] = Tri(0, 2, float3(0, 1, 0), float3(1,0,0), float3(0));		// 6: floor
		objects[1] = Tri(1, 2, float3(1, 0, 0), float3(3,0,0), float3(0));		// 4: left wall
		objects[2] = Tri(2, 2, float3(-1, 0, 0), float3(3,0,0), float3(0));		// 5: right wall
		objects[3] = Tri(3, 2, float3(0, 0, 1), float3(3,0,0), float3(0));		// 9: back wall
		/*
		objects[4] = Tri(3, 2, float3(0, 0, -1), float3(3,0,0), float3(0));		// 9: back wall
		objects[4] = Tri(4, 1, float3(0, 0, 2), float3(0.5f), float3(0));

		objects[5] = Tri(5, 0, float3(0,1,1), float3(-0.5f, 1.5f, 1), float3(0.5f, 1.5f, 1));
		*/

		for (int i = 4; i < N; i++)
		{
			float3 r0(RandomFloat(), RandomFloat(), RandomFloat());
			float3 r1(RandomFloat(), RandomFloat(), RandomFloat());
			float3 r2(RandomFloat(), RandomFloat(), RandomFloat());
			if (r2.x < 0.5) objects[i] = Tri( i, 1, r0 * 3, float3(0.5f), float3(0) );
			else objects[i] = Tri( i, 0, r0 * 2 - float3(1), r0 * 2 - float3(1) + r1, r0 * 2 - float3(1) + r2 );
		}
		
		//bvh = BVH(objects, nr_of_objects);
		BuildBVH();
		printBVH();
		collapseBVH(rootNodeIdx);
		printBVH();
		SetTime(0);
	}

	void SetTime(float t)
	{
		// default time for the scene is simply 0. Updating/ the time per frame 
		// enables animation. Updating it per ray can be used for motion blur.
		animTime = t;
		
		// sphere animation: bounce
		//float tm = 1 - sqrf(fmodf(animTime, 2.0f) - 1);
		//objects[4].pos = float3(0, -0.5f + tm, 2);
	}
	float3 GetLightPos() const
	{
		return (light.v0 + light.v1 + light.v2) / 3;
	}
	float3 GetLightColor() const
	{
		return float3(2, 2, 2);
	}
	void FindNearest(Ray& ray)
	{
		//for (int i = 0; i < nr_of_objects; i++) { objects[i].Intersect(ray); }
		IntersectBVH(ray, rootNodeIdx);
	}
	bool IsOccluded(Ray& ray)
	{
		float rayLength = ray.t;
		//for (int i = 0; i < nr_of_objects; i++) { if (objects[i].shapeType != 2) objects[i].Intersect(ray); }
		// skip root's left node to skip occlusion check for infinite primitives
		IntersectBVH(ray, 2, true);
		return ray.t < rayLength;
	}
	float3 GetNormal(int objIdx, float3 I, float3 wo) const
	{
		//if (objIdx == 6) return T.GetNormal(I);
		return objects[objIdx].GetNormal(I);
	}
	float3 GetAlbedo(int objIdx, float3 I) const
	{
		//if (objIdx == 6) return T.GetNormal(I);
		return objects[objIdx].GetAlbedo(I);
	}
	float GetReflectivity(int objIdx, float3 I) const
	{
		if (objIdx == 1 /* ball */) return 1;
		if (objIdx == 6 /* floor */) return 0.3f;
		return 0;
	}
	float GetRefractivity(int objIdx, float3 I) const
	{
		return objIdx == 3 ? 1.0f : 0.0f;
	}
	Material GetMaterial(int objIdx) const
	{
		return objects[objIdx].material;
	}

	void BuildBVH()
	{
		for (int i = 0; i < N; i++) triIdx[i] = i;

		// assign all infinite primitives to root's left child
		BVHNode& root = bvhNode[rootNodeIdx];
		root.leftNode = 1;
		root.rightNode = 2;
		root.firstTriIdx = root.triCount = 0;
		int i = root.firstTriIdx;
		int j = i + N - 1;
		while (i <= j)
		{
			if (objects[triIdx[i]].shapeType == 2) { i++; bvhNode[rootNodeIdx + 1].triCount++; }
			else swap(triIdx[i], triIdx[j--]);
		}
		root.aabbMin = bvhNode[root.leftNode].aabbMin = float3(1e30f);
		root.aabbMax = bvhNode[root.leftNode].aabbMax = float3(-1e30f);
		bvhNode[root.leftNode].firstTriIdx = 0;

		// assign finite primites to root's right child
		bvhNode[rootNodeIdx + 2].firstTriIdx = bvhNode[root.leftNode].triCount;
		bvhNode[rootNodeIdx + 2].triCount = N - bvhNode[root.leftNode].triCount;
		UpdateNodeBounds(rootNodeIdx + 2);

		// subdivide  finite primites recursively
		Subdivide(rootNodeIdx + 2);
	}


	void UpdateNodeBounds(uint nodeIdx)
	{
		BVHNode& node = bvhNode[nodeIdx];
		node.aabbMin = float3(1e30f);
		node.aabbMax = float3(-1e30f);
		for (uint first = node.firstTriIdx, i = 0; i < node.triCount; i++)
		{
			Tri& leafTri = objects[triIdx[first + i]];
			switch (leafTri.shapeType) {
			case 0:
				node.aabbMin = fminf(node.aabbMin, leafTri.v0);
				node.aabbMin = fminf(node.aabbMin, leafTri.v1);
				node.aabbMin = fminf(node.aabbMin, leafTri.v2);
				node.aabbMax = fmaxf(node.aabbMax, leafTri.v0);
				node.aabbMax = fmaxf(node.aabbMax, leafTri.v1);
				node.aabbMax = fmaxf(node.aabbMax, leafTri.v2);
				break;
			case 1:
				node.aabbMin = fminf(node.aabbMin, leafTri.pos - float3(sqrt(leafTri.r2)));
				node.aabbMax = fmaxf(node.aabbMax, leafTri.pos + float3(sqrt(leafTri.r2)));
				break;
			case 2:
				// planes are infinite
				break;
			}
		}
	}

	void Subdivide(uint nodeIdx)
	{
		// terminate recursion
		BVHNode& node = bvhNode[nodeIdx];
		if (node.triCount <= 2) return;
		// determine split axis and position
		float3 extent = node.aabbMax - node.aabbMin;
		int axis = 0;
		if (extent.y > extent.x) axis = 1;
		if (extent.z > extent[axis]) axis = 2;
		float splitPos = node.aabbMin[axis] + extent[axis] * 0.5f;
		// in-place partition
		int i = node.firstTriIdx;
		int j = i + node.triCount - 1;
		while (i <= j)
		{
			if (objects[triIdx[i]].centroid[axis] < splitPos)
				i++;
			else
				swap(triIdx[i], triIdx[j--]);
		}
		// abort split if one of the sides is empty
		int leftCount = i - node.firstTriIdx;
		if (leftCount == 0 || leftCount == node.triCount) return;
		// create child nodes
		int leftChildIdx = nodesUsed++;
		int rightChildIdx = nodesUsed++;
		node.leftNode = leftChildIdx;
		node.rightNode = rightChildIdx; // added for collapsing purposes
		bvhNode[leftChildIdx].firstTriIdx = node.firstTriIdx;
		bvhNode[leftChildIdx].triCount = leftCount;
		bvhNode[rightChildIdx].firstTriIdx = i;
		bvhNode[rightChildIdx].triCount = node.triCount - leftCount;
		node.triCount = 0;
		UpdateNodeBounds(leftChildIdx);
		UpdateNodeBounds(rightChildIdx);
		// recurse
		Subdivide(leftChildIdx);
		Subdivide(rightChildIdx);
	}

	void IntersectBVH(Ray& ray, const uint nodeIdx, bool checkOcclusion = false)
	{
		BVHNode& node = bvhNode[nodeIdx];
		//if (nodeIdx == 1) { for (uint i = 0; i < node.triCount; i++)  objects[triIdx[node.firstTriIdx + i]].Intersect(ray); }
		if (!IntersectAABB(ray, node.aabbMin, node.aabbMax)) return;
		if (node.isLeaf())
		{
			if (checkOcclusion) {
				float rayLength = ray.t;
				for (uint i = 0; i < node.triCount; i++) {
					objects[triIdx[node.firstTriIdx + i]].Intersect(ray);
					if (ray.t < rayLength) return;
				}

			}
			else {
				for (uint i = 0; i < node.triCount; i++)
					objects[triIdx[node.firstTriIdx + i]].Intersect(ray);
			}
		}
		//else
		//{
			if (node.leftNode != 0) {
				IntersectBVH(ray, node.leftNode);
				IntersectBVH(ray, node.rightNode);
			}
		//}
	}

	bool IntersectAABB(const Ray& ray, const float3 bmin, const float3 bmax) const
	{
		float tx1 = (bmin.x - ray.O.x) / ray.D.x, tx2 = (bmax.x - ray.O.x) / ray.D.x;
		float tmin = min(tx1, tx2), tmax = max(tx1, tx2);
		float ty1 = (bmin.y - ray.O.y) / ray.D.y, ty2 = (bmax.y - ray.O.y) / ray.D.y;
		tmin = max(tmin, min(ty1, ty2)), tmax = min(tmax, max(ty1, ty2));
		float tz1 = (bmin.z - ray.O.z) / ray.D.z, tz2 = (bmax.z - ray.O.z) / ray.D.z;
		tmin = max(tmin, min(tz1, tz2)), tmax = min(tmax, max(tz1, tz2));
		return tmax >= tmin && tmin < ray.t&& tmax > 0;
	}

	void collapseBVH(uint nodeIdx) {
		cout << "collapsing node " << nodeIdx << "\n";
		BVHNode& node = bvhNode[nodeIdx];
		if (!node.isLeaf()) {
			collapseBVH(node.leftNode);
			collapseBVH(node.rightNode);

			// check if we can collapse children of node
			BVHNode left = bvhNode[bvhNode[nodeIdx].leftNode];
			int lc = left.isLeaf() ? left.triCount : 2;
			bool can_collapse_left = 2 - 1 + lc <= 4;

			BVHNode right = bvhNode[bvhNode[nodeIdx].rightNode];
			int rc = right.isLeaf() ? right.triCount : 2;
			bool can_collapse_right = 2 - 1 + rc <= 4;

			if ( ! ( can_collapse_left || can_collapse_right ) ) return;

			// select child of node with greatest area
			float3 extend_left = left.aabbMax - left.aabbMin;
			float3 extend_right = right.aabbMax - right.aabbMin;
			bool choose_left = extend_left.x*extend_left.y*extend_left.z >= extend_right.x* extend_right.y* extend_right.z;

			// replace node with child
			if (choose_left && can_collapse_left) {
				if (left.isLeaf()) {
					node.firstTriIdx = left.firstTriIdx;
					node.leftNode = left.leftNode;
					node.triCount += left.triCount;
				}
				else {
					node.leftNode = left.leftNode;
				}
			
			}
			else if (can_collapse_right) {
				if (right.isLeaf()) {
					node.firstTriIdx = right.firstTriIdx;
					node.triCount += right.triCount;
				}
				else {
					node.rightNode = right.leftNode;
				}
			}


		}
	}

	void printBVH(uint start = 0, int spaces = 0) {
		BVHNode& node = bvhNode[start];
		nSpaces(spaces);
		cout << start << " ";
		printf("min: %.2f, %.2f, %.2f\n", node.aabbMin.x, node.aabbMin.y, node.aabbMin.z);
		spaces += 2;
		nSpaces(spaces);
		printf("max: %.2f, %.2f, %.2f\n", node.aabbMax.x, node.aabbMax.y, node.aabbMax.z);
		if (node.isLeaf()) {
			nSpaces(spaces);
			cout << "LEAF TRICOUNT: "  << node.triCount << "\n";
		}

		if (node.leftNode != 0) printBVH(node.leftNode, spaces);
		if (node.rightNode != 0) printBVH(node.rightNode, spaces);


		if (start == 0) cout << "BVH Depth:" << BVHDepth(rootNodeIdx) << "\n";
		if (start == 0) cout << "BVH Nodes:" << BVHNodes(rootNodeIdx) << "\n";
		if (start == 0) cout << "BVH Area: " << BVHAreaSummed(bvhNode[rootNodeIdx].rightNode) << "\n";

	}

	void nSpaces(int n) {
		for (int i = 0; i < n; i++) cout << " ";
	}

	int BVHDepth(uint start) {
		BVHNode& node = bvhNode[start];
		int l = 0;
		int r = 0;
		if (node.leftNode != 0) l = BVHDepth(node.leftNode) + 1;
		if (node.rightNode != 0) r = BVHDepth(node.rightNode) + 1;
		return max(l, r);
	}

	int BVHNodes(uint start) {
		BVHNode& node = bvhNode[start];
		int l = 0;
		int r = 0;
		if (node.leftNode != 0) l = BVHNodes(node.leftNode);
		if (node.rightNode != 0) r = BVHNodes(node.rightNode);
		return 1 + l + r;
	}

	float BVHAreaSummed(uint start) {
		BVHNode& node = bvhNode[start];
		if (node.isLeaf()) return (node.aabbMax.x - node.aabbMin.x) * (node.aabbMax.y - node.aabbMin.y) * (node.aabbMax.z - node.aabbMin.z);
		return BVHAreaSummed(node.leftNode) + BVHAreaSummed(node.rightNode);
	}

	struct BVHNode
	{
		float3 aabbMin, aabbMax;
		uint leftNode, firstTriIdx, triCount;
		uint rightNode; // added for collapsing purposes
		bool isLeaf() { return triCount > 0; }
	};

	// N * 2 - 1 -- with N the number of objects -- is the maximum of BVH nodes
	int N = 16;
	uint triIdx[16];
	BVHNode bvhNode[16 * 2 - 1];
	uint rootNodeIdx = 0, nodesUsed = 3;

	__declspec(align(64)) // start a new cacheline here
		float animTime = 0;
	Tri light;
	int nr_of_objects = 16;
	Tri objects[16];
};

}