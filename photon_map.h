#pragma once
#include <concepts>
#include <numeric>
#include <queue>

// reference: https://github.com/yumcyaWiz/photon_mapping/blob/main/include/photon_map.h
// reference: https://www.cs.uu.nl/docs/vakken/magr/2022-2023/slides/lecture%2013%20-%20bidirectional.pdf p.35

struct Photon {
	float3 position;	// world space position of the photon hit
	float3 power;		// current power level for the photon
	float3 L;			// incident direction

	// implementation of Point concept
	static constexpr int dim = 3;
	float operator[](int i) const { return position.cell[i]; }

	Photon() {}
	Photon(const float3& position, const float3& power, const float3& L)
		: position(position), power(power), L(L) {}
};

struct Node
{
	int axis;
	int idx;
	int leftChildIdx;
	int rightChildIdx;

	Node() : axis(-1), idx(-1), leftChildIdx(-1), rightChildIdx(-1) {}
};

class PhotonKDTree
{
private:
	std::vector<Node> nodes;
	std::vector<int> indices;
	const Photon* points;
	int points_count;

	void createNode(int* indices, int indices_count, int depth)
	{
		if (indices_count <= 0) return;

		const int axis = depth % 3; // looping x, y, and z

		std::sort(indices, indices + indices_count, [&](int idx1, int idx2) {
			return points[idx1][axis] < points[idx2][axis];
		});

		const int middle = (indices_count - 1) / 2;

		const int parentIdx = nodes.size();
		Node node;
		node.axis = axis;
		node.idx = indices[middle];
		nodes.push_back(node);

		const int leftChildIdx = nodes.size();
		createNode(indices, middle, depth + 1);

		if (leftChildIdx == nodes.size())
		{
			nodes[parentIdx].leftChildIdx = -1;
		}
		else

		{
			nodes[parentIdx].leftChildIdx = leftChildIdx;
		}

		const int rightChildIdx = nodes.size();
		createNode(indices + middle + 1, indices_count - middle - 1, depth + 1);

		if (rightChildIdx == nodes.size())
		{
			nodes[parentIdx].rightChildIdx = -1;
		}
		else
		{
			nodes[parentIdx].rightChildIdx = rightChildIdx;
		}
	}

	using KNNQueue = std::priority_queue<std::pair<float, int>>;
	void searchKNearestNode(int nodeIdx, const float3& queryPoint, int k,
		KNNQueue& queue) const 
	{
		if (nodeIdx < 0 || nodeIdx >= nodes.size()) return;

		const Node& node = nodes[nodeIdx];

		const Photon median = points[node.idx];

		const float dist = length(queryPoint - median.position);
		queue.emplace(dist, node.idx);

		if (queue.size() > k)
		{
			queue.pop();
		}

		const bool isLower = queryPoint.cell[node.axis] < median[node.axis];
		if (isLower)
		{
			searchKNearestNode(node.leftChildIdx, queryPoint, k, queue);
		}
		else

		{
			searchKNearestNode(node.rightChildIdx, queryPoint, k, queue);
		}

		// at leaf node, if size of queue is smaller than k, or queue's largest
		// minimum distance overlaps sibblings region, then search siblings
		const float dist_to_siblings = median[node.axis] - queryPoint.cell[node.axis];
		if (queue.top().first > (dist_to_siblings * dist_to_siblings))
		{
			if (isLower)
			{
				searchKNearestNode(node.rightChildIdx, queryPoint, k, queue);
			}
			else

			{
				searchKNearestNode(node.leftChildIdx, queryPoint, k, queue);
			}
		}
	}
public:

	void setPoints(const Photon* points, int points_count)
	{
		this->points = points;
		this->points_count = points_count;
	}

	void buildTree()
	{
		std::vector<int> indices(points_count);
		
		for (int i = 0; i < points_count; i++)
		{
			indices[i] = i;
		}

		createNode(indices.data(), points_count, 0);
	}

	std::vector<int> searchKNearest(
		const float3 queryPoint, int k, float& maxDist2) const
	{
		KNNQueue queue;
		searchKNearestNode(0, queryPoint, k, queue);

		std::vector<int> ret(queue.size());
		maxDist2 = 0;

		for (int i = 0; i < ret.size(); i++)
		{
			const auto& p = queue.top();
			ret[i] = p.second;
			maxDist2 = max(maxDist2, p.first);
			queue.pop();
		}

		return ret;
	}
};

class PhotonMap
{
private:
	std::vector<Photon> photons;
	PhotonKDTree kdTree;
public:
	const int getPhotonCount() const { return photons.size(); }
	const Photon& getPhoton(int idx) const { return photons[idx]; }

	void addPhoton(const Photon& photon) { photons.push_back(photon); }
	void setPhotons(const std::vector<Photon>& photons)
	{
		this->photons = photons;
	}

	void build()
	{
		kdTree.setPoints(photons.data(), photons.size());
		kdTree.buildTree();
	}

	std::vector<int> queryKNearestPhotons(const float3 queryPoint, int k,
		float& max_dist2) const
	{
		return kdTree.searchKNearest(queryPoint, k, max_dist2);
	}
};