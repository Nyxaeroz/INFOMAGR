#include "precomp.h"

// -----------------------------------------------------------
// Initialize the renderer
// -----------------------------------------------------------
void Renderer::Init()
{
	// create fp32 rgb pixel buffer to render to
	accumulator = (float4*)MALLOC64( SCRWIDTH * SCRHEIGHT * 16 );
	memset( accumulator, 0, SCRWIDTH * SCRHEIGHT * 16 );
}

// -----------------------------------------------------------
// Evaluate light transport
// -----------------------------------------------------------
float3 Renderer::Trace( Ray& ray, int depth = 0)
{
	if (depth > 10) return float3(1);
	depth++;
	
	scene.FindNearest( ray );
	if (ray.objIdx == -1) return 0; // or a fancy sky color
	float3 I = ray.O + ray.t * ray.D;
	float3 N = scene.GetNormal( ray.objIdx, I, ray.D );
	float3 albedo = scene.GetAlbedo( ray.objIdx, I );
	Mat mat = scene.GetMaterial(ray.objIdx).type;

	if (scene.IsOccluded(ray)) return float3(0);
	else if (mat == Mat::DIFFUSE) {
		float3 colorScale = float3( 0 );
		directIllumination( I, N, colorScale );
		return albedo * colorScale;
	}
	else if (mat == Mat::MIRROR) {
		float3 reflectedDir = reflect(ray.D, N); //already unit length
		Ray reflectedRay = Ray(I + EPSILON * reflectedDir, reflectedDir);
		return albedo * Trace(reflectedRay, depth);
	}
	else if (mat == Mat::GLASS) {
		float3 reflectedDir = reflect(ray.D, N); //already unit length
		Ray reflectedRay = Ray(I + EPSILON * reflectedDir, reflectedDir);
		
		float n1 = 1.00; 
		float n2 = 1.52; 
		float n = n1 / n2;
		float c1 = -dot(N, ray.D);
		float k = 1 - n * n * (1 - c1 * c1);
		float3 T;
		if (k < 0) return albedo * Trace(reflectedRay, depth);
		else 
		{
			T = n * ray.D + N * (n * c1 - sqrt(k));
			Ray refractedRay = Ray(I + EPSILON * T, normalize(T));
			
			/* Without Beer's Law:
			return 0.9 * albedo * Trace(refractedRay, depth)
				 + 0.1 * albedo * Trace(reflectedRay, depth);
			*/

			/* Beer's Law attempt */
			Ray inMaterialRay = Ray(I + float3(0.1) * ray.D, ray.D);
			scene.FindNearest(inMaterialRay);
			float3 I2 = inMaterialRay.O + inMaterialRay.t * inMaterialRay.D;
			float d = 0;
			if (ray.objIdx == inMaterialRay.objIdx) float d = length(I2 - I);
			float3 a = (0.3);
			float3 intensity;
			intensity.x = exp(-a.x * d);
			intensity.y = exp(-a.y * d);
			intensity.z = exp(-a.z * d);

			return 0.9 * albedo * Trace(refractedRay, depth) * intensity
				 + 0.1 * albedo * Trace(reflectedRay, depth);

		}
	}
	else return float3(0);
	
	
	/* visualize normal */ // return (N + 1) * 0.5f;
	/* visualize distance */  // return 0.1f * float3( ray.t, ray.t, ray.t );
	/* visualize albedo */  // return albedo;
}

float3 Renderer::TracePath(Ray& ray, int depth = 0)
{
	if (depth > 10) return float3(0);
	depth++;

	scene.FindNearest(ray);
	if (ray.objIdx == -1) return 0;
	if (ray.objIdx == 0) return scene.GetLightColor();
	Mat mat = scene.GetMaterial(ray.objIdx).type;
	float3 I = ray.O + ray.t * ray.D;
	float3 N = scene.GetNormal(ray.objIdx, I, ray.D);
	float3 albedo = scene.GetAlbedo(ray.objIdx, I);
	if (mat == Mat::MIRROR) {
		float3 reflectedDir = reflect(ray.D, N); //already unit length
		Ray reflectedRay = Ray(I + EPSILON * reflectedDir, reflectedDir);
		return albedo * TracePath(reflectedRay, depth);
	}
	if (mat == Mat::GLASS) {
		if (RandomFloat() < 0.9) {
			float n1 = 1.00;
			float n2 = 1.52;
			float n = n1 / n2;
			float c1 = -dot(N, ray.D);
			float k = 1 - n * n * (1 - c1 * c1);
			if (k >= 0) {
				float3 T;
				T = n * ray.D + N * (n * c1 - sqrt(k));
				Ray refractedRay = Ray(I + EPSILON * T, normalize(T));
				return albedo * TracePath(refractedRay, depth);
			}
		}
		float3 reflectedDir = reflect(ray.D, N); //already unit length
		Ray reflectedRay = Ray(I + EPSILON * reflectedDir, reflectedDir);
		return albedo * TracePath(reflectedRay, depth);
		

	}

	float3 R = randomHemDir(N);
	Ray rayToHemisphere = Ray(I + R * EPSILON, R);
	float3 BRDF = albedo * INVPI;
	float3 Ei = TracePath(rayToHemisphere, depth) * dot(N,R);
	return 2.0f * PI * BRDF * Ei * 0.5;
}


float3 Renderer::randomHemDir(float3 N)
{
	// create vector in (-1,-1,-1) - (1,1,1) cube
	float3 R;
	do {
		R.x = RandomFloat();
		R.y = RandomFloat();
		R.z = RandomFloat();
	} while (length(R) > 1);

	// if R points to the other side of a surface N is normal to, flip it
	if (dot(N, R) < 0) R = -R;

	return normalize(R);

}

void Renderer::directIllumination(float3 I, float3 N, float3& colorScale)
{
	float3 lightPos = scene.GetLightPos();
	float3 shadowRayDir = normalize(lightPos - I);
	Ray shadowRay = Ray(I + float3(0.001) * shadowRayDir, shadowRayDir);
	scene.light.Intersect(shadowRay);

	if (!scene.IsOccluded(shadowRay)) {
		float distToLight = length(lightPos - I);
		float distFactor = 1 / (distToLight * distToLight);
		float angleFactor = dot(N, lightPos - I);
		colorScale.x = scene.GetLightColor().x * distFactor * angleFactor;
		colorScale.y = scene.GetLightColor().y * distFactor * angleFactor;
		colorScale.z = scene.GetLightColor().z * distFactor * angleFactor;
	}
}

float Renderer::Fresnel(float n1, float n2, float3 N, float3 D)
{
	return 1;
}

// -----------------------------------------------------------
// Main application tick function - Executed once per frame
// -----------------------------------------------------------
void Renderer::Tick( float deltaTime )
{
	Timer t;
	// make one check on using whitted or path tracing. Ugly result, but reduces operations
	if (path) {
		// animation
		scene.SetTime(0);
		// pixel loop
		// lines are executed as OpenMP parallel tasks (disabled in DEBUG)
		#pragma omp parallel for schedule(dynamic)
		for (int y = 0; y < SCRHEIGHT; y++)
		{
			// trace a primary ray for each pixel on the line
			for (int x = 0; x < SCRWIDTH; x++) {
				float3 pt = TracePath(camera.GetPrimaryRay(x, y));
				if (length(pt) < EPSILON && length(accumulator[x + y * SCRWIDTH]) > EPSILON);
				else accumulator[x + y * SCRWIDTH] = float4(pt, 0);
			}

			// translate accumulator contents to rgb32 pixels
			for (int dest = y * SCRWIDTH, x = 0; x < SCRWIDTH; x++)
				screen->pixels[dest + x] =
				RGBF32_to_RGB8(&accumulator[x + y * SCRWIDTH]);
		}
	}
	else {
		// animation
		static float animTime = 0;
		scene.SetTime(animTime+= deltaTime * 0.002f);
		// pixel loop
		// camera movement 
		float3 v1 = camera.bottomLeft - camera.topLeft;
		float3 v2 = camera.viewDir;
		float3 n = normalize(camera.crossProd(v1, v2));
		if (movingW) camera.updateCamPos(camera.viewDir / 30);
		if (movingA) camera.updateCamPos(n / 30);
		if (movingS) camera.updateCamPos(-camera.viewDir / 30);
		if (movingD) camera.updateCamPos(-n / 30);

		// lines are executed as OpenMP parallel tasks (disabled in DEBUG)
		#pragma omp parallel for schedule(dynamic)
		for (int y = 0; y < SCRHEIGHT; y++)
		{
			// trace a primary ray for each pixel on the line
			for (int x = 0; x < SCRWIDTH; x++)
				accumulator[x + y * SCRWIDTH] = float4(Trace( camera.GetPrimaryRay( x, y ) ), 0);
				
			// translate accumulator contents to rgb32 pixels
			for (int dest = y * SCRWIDTH, x = 0; x < SCRWIDTH; x++)
				screen->pixels[dest + x] = 
					RGBF32_to_RGB8( &accumulator[x + y * SCRWIDTH] );
		}
	}
	// performance report - running average - ms, MRays/s
	static float avg = 10, alpha = 1;
	avg = (1 - alpha) * avg + alpha * t.elapsed() * 1000;
	if (alpha > 0.05f) alpha *= 0.5f;
	float fps = 1000 / avg, rps = (SCRWIDTH * SCRHEIGHT) * fps;
	//printf( "%5.2fms (%.1fps) - %.1fMrays/s\n", avg, fps, rps / 1000000 );
}