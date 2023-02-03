#pragma once

// default screen resolution
#define SCRWIDTH	1280
#define SCRHEIGHT	720
// #define FULLSCREEN
// #define DOUBLESIZE

namespace Tmpl8 {

class Camera
{
public:
	Camera()
	{
		// setup a basic view frustum
		camPos = float3( 0, 0, -2 );
		viewDir = float3(0, 0, 1);
		topLeft = float3( -aspect, 1, 0 );
		topRight = float3( aspect, 1, 0 );
		bottomLeft = float3( -aspect, -1, 0 );
	}
	void updateCamPos(float3 d)
	{
		camPos += d;
		topLeft += d;
		topRight += d;
		bottomLeft += d;
	}
	void updateFOV(float d)
	{
		if (d > 0 && FOV < 1.9) FOV += d;
		else if (d < 0 && FOV > 0.2) FOV += d;
		/*
		topLeft = topLeft - (topRight - topLeft) * d;
		topRight = topRight - (topLeft - topRight) * d;
		bottomLeft = bottomLeft - (topRight - topLeft) * d;
		*/

		cout << FOV << "\n";
	}
	void rotateLocalX(float th)
	{
		float3 axis = normalize(topLeft - topRight);
		topLeft    = camPos + rotAboutVector(topLeft - camPos, axis, th);
		topRight   = camPos + rotAboutVector(topRight - camPos, axis, th);
		bottomLeft = camPos + rotAboutVector(bottomLeft - camPos, axis, th);

		viewDir    = rotAboutVector(viewDir, axis, th);
	}
	void rotateLocalY(float th)
	{
		float3 axis = normalize(bottomLeft - topLeft);
		topLeft    = camPos + rotAboutVector(topLeft - camPos, axis, th);
		topRight   = camPos + rotAboutVector(topRight - camPos, axis, th);
		bottomLeft = camPos + rotAboutVector(bottomLeft - camPos, axis, th);

		viewDir	   = rotAboutVector(viewDir, axis, th);
	}

	/* Rotate point p about vector u with angle th

		Using axis angle rotation:

				|  0   -uz   uy |
		Let C = |  uz   0   -ux |
				| -uy   ux   0  |

		Rotation matrix: R_u(th) = I + C * sin(th) + C^2(1 - cos(th))
	*/
	float3 rotAboutVector(float3 p, float3 u, float th)
	{
		return float3(
			p.x * (1-(1-cos(th))*(u.z*u.z+u.y*u.y)) + p.y * (-u.z*sin(th) + u.x*u.y*(1-cos(th))) + p.z * (u.y*sin(th) + u.x*u.y*(1-cos(th))),
			p.x * (u.z*sin(th) + u.x*u.y*(1-cos(th))) + p.y * (1-(1-cos(th))*(u.z*u.z+u.x*u.x)) + p.z * (-u.x*sin(th) + u.y*u.z*(1-cos(th))),
			p.x * (-u.y*sin(th) + u.x*u.z*(1-cos(th))) + p.y * (u.x*sin(th) + u.y*u.z*(1-cos(th))) + p.z * (1-(1-cos(th))*(u.x*u.x+u.y*u.y))
		);
	}

	float3 crossProd(float3 a, float3 b)
	{
		return float3(
			a.y*b.z - a.z*b.y,
			a.z*b.x - a.x*b.z,
			a.x*b.y - a.y*b.x
		);
	}

	Ray GetPrimaryRay( const int x, const int y )
	{
		// calculate pixel position on virtual screen plane
		const float u = (float)x * (1.0f / SCRWIDTH);
		const float v = (float)y * (1.0f / SCRHEIGHT);

		// use FOV to determine screencorners
		float3 ntopLeft    = topLeft    + (topRight - topLeft + bottomLeft - topLeft) * (1 - FOV);
		float3 ntopRight   = topRight   + (topLeft - topRight + bottomLeft - topLeft) * (1 - FOV);
		float3 nbottomLeft = bottomLeft + (topRight - topLeft - bottomLeft + topLeft) * (1 - FOV);

		const float3 P = ntopLeft + u * (ntopRight - ntopLeft) + v * (nbottomLeft - ntopLeft);
		return Ray( camPos, normalize( P - camPos ) );
	}

	void setDirty(bool v)
	{
		dirty = v;
	}
	float aspect = (float)SCRWIDTH / (float)SCRHEIGHT;
	float3 camPos, viewDir, screenCenter;
	float FOV = 1;
	float3 topLeft, topRight, bottomLeft;
	bool dirty = false;
};

}