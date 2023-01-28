#pragma once

namespace Tmpl8
{

class Renderer : public TheApp
{
public:
	// game flow methods
	void Init();
	float3 Trace( Ray& ray, int depth );
	float3 TracePath( Ray& ray, int depth );
	float3 ShowPhotons( Ray& ray );
	float3 PhotonPath( Ray& ray );
	void CreatePhotonMap();
	float3 randomHemDir( float3 N );
	void directIllumination( float3 I, float3 N, float3& colorScale );
	float Fresnel(float n1, float n2, float3 N, float3 D);
	void Tick( float deltaTime );
	void Shutdown() { /* implement if you want to do something on exit */ }
	
	// -----------------------------------------------------------
	// Input Handling Mouse
	// -----------------------------------------------------------
	void MouseUp(int button) { dragging = false; }
	void MouseDown(int button) { dragging = true; }
	void MouseMove(int x, int y) {
		if (dragging) {
			camera.rotateLocalY((x - mousePos.x) / 1000.0);
			camera.rotateLocalX((y - mousePos.y) / 1000.0);
		}
		mousePos.x = x, mousePos.y = y;
	}
	void MouseWheel(float y) { 
		camera.updateFOV(-y/10);
	}
	void KeyUp( int key ) { 
		switch (key) {
		case 87: // w
			movingW = false;
			break;
		case 65: // a
			movingA = false;
			break;
		case 83: // s
			movingS = false;
			break;
		case 68: // d
			movingD = false;
			break;
		}
	}
	void KeyDown( int key ) { 
		switch (key) {
		case 87: // w
			movingW = true;
			break;
		case 65: // a
			movingA = true;
			break;
		case 83: // s
			movingS = true;
			break;
		case 68: // d
			movingD = true;
			break;
		}
	}
	// data members
	int2 mousePos;
	float4* accumulator;
	Camera camera;
	bool dragging = false;
	bool movingW, movingA, movingS, movingD = false;
	const float EPSILON = 0.01;

	Scene scene;
	bool path = true; // quick flag for whitted (false) or path (true) tracing

	PhotonMap photonmap = PhotonMap();
};

} // namespace Tmpl8