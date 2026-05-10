#include <iostream>
#include <vector>
#include <queue>
#include <climits>
#include <algorithm>
#include <set>
#include <functional>
#include <random>
#include <stdexcept>
#include <map>
#include <stack>
#include <list>
#include <omp.h>
#include <mpi.h>
#include <numeric>

using namespace std;

const int MAX_OMP_THREADS = 8;
const int ROOT_RANK = 0;

class Matrix {
protected:
    std::vector<std::vector<int>> data;
    int rows, cols;

public:
    Matrix(int r, int c) : rows(r), cols(c) {
        data.resize(rows, std::vector<int>(cols, 0));
    }

    virtual ~Matrix() = default;

    int getRows() const { return rows; }
    int getCols() const { return cols; }

    int get(int row, int col) const {
        if (row >= 0 && row < rows && col >= 0 && col < cols) {
            return data[row][col];
        }
        throw std::out_of_range("Invalid index");
    }

    void set(int row, int col, int value) {
        if (row >= 0 && row < rows && col >= 0 && col < cols) {
            data[row][col] = value;
        } else {
            throw std::out_of_range("Invalid index");
        }
    }

    void print() const {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                std::cout << data[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    void printWeightMatrix() const {
        std::cout << "Weight matrix:\n";
        print();
    }
};

class Graph : public Matrix {
public:
    Graph(int vertices) : Matrix(vertices, vertices) {}

    virtual void generateGraph(bool allowNegativeWeights) = 0;
    virtual void printInfo() const = 0;
};

class DirectedGraph : public Graph {
private:
    void syncMatrixAcrossProcesses() {
        int n = getRows();
        std::vector<int> flatMatrix(n * n);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                flatMatrix[i * n + j] = get(i, j);
            }
        }

        MPI_Allreduce(MPI_IN_PLACE, flatMatrix.data(), n * n, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                set(i, j, flatMatrix[i * n + j]);
            }
        }
    }

public:
    DirectedGraph(int vertices) : Graph(vertices) {}
    void generateGraph(bool allowNegativeWeights) override {
        std::random_device rd;
        std::mt19937 gen(rd());
        int minW = allowNegativeWeights ? -10 : 1;
        std::uniform_int_distribution<int> dist(minW, 10);

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < getRows(); ++i) {
            for (int j = 0; j < getCols(); ++j) {
                if (i != j && dist(gen) > 5) {
                    #pragma omp critical
                    set(i, j, dist(gen));
                }
            }
        }
    }

    void generateGraphDistributed(bool allowNegativeWeights, int world_rank, int world_size) {
        std::random_device rd;
        std::mt19937 gen(rd());
        int minW = allowNegativeWeights ? -10 : 1;
        std::uniform_int_distribution<int> dist(minW, 10);

        int n = getRows();
        int chunk_size = n / world_size;
        int start = world_rank * chunk_size;
        int end = (world_rank == world_size - 1) ? n : start + chunk_size;

        #pragma omp parallel for collapse(2)
        for (int i = start; i < end; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    #pragma omp critical
                    set(i, j, dist(gen));
                }
            }
        }

        syncMatrixAcrossProcesses();
    }

    void printInfo() const override {
        std::cout << "Directed Graph with " << getRows() << " vertices\n";
        printWeightMatrix();
    }

    struct BellmanFordResult {
        bool hasNegativeCycle;
        int iterations;
        std::vector<int> distances;
        std::vector<int> predecessors;
    };

    BellmanFordResult bellmanFord(int source) const {
        int n = getRows();
        BellmanFordResult result;
        result.iterations = 0;
        result.distances.assign(n, INT_MAX);
        result.predecessors.assign(n, -1);
        result.hasNegativeCycle = false;
        result.distances[source] = 0;

        for (int i = 0; i < n - 1; ++i) {
            bool updated = false;

            #pragma omp parallel for reduction(||:updated)
            for (int u = 0; u < n; ++u) {
                if (result.distances[u] == INT_MAX) continue;

                for (int v = 0; v < n; ++v) {
                    int weight = get(u, v);
                    if (weight != 0 && result.distances[u] != INT_MAX) {
                        if (result.distances[v] > result.distances[u] + weight) {
                            #pragma omp critical
                            {
                                result.distances[v] = result.distances[u] + weight;
                                result.predecessors[v] = u;
                                updated = true;
                            }
                        }
                    }
                }
            }
            result.iterations++;

            if (!updated) break;
        }

        #pragma omp parallel for
        for (int u = 0; u < n; ++u) {
            if (result.distances[u] == INT_MAX) continue;

            for (int v = 0; v < n; ++v) {
                int weight = get(u, v);
                if (weight != 0 && result.distances[u] != INT_MAX) {
                    if (result.distances[v] > result.distances[u] + weight) {
                        result.hasNegativeCycle = true;
                        result.distances[v] = INT_MIN;
                    }
                }
            }
        }

        return result;
    }

    BellmanFordResult bellmanFordDistributed(int source, int world_rank, int world_size) const {
        int n = getRows();
        BellmanFordResult result;
        result.iterations = 0;
        result.distances.assign(n, INT_MAX);
        result.predecessors.assign(n, -1);
        result.hasNegativeCycle = false;

        if (world_rank == 0) {
            result.distances[source] = 0;
        }

        MPI_Bcast(result.distances.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(result.predecessors.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

        int chunk_size = n / world_size;
        int remainder = n % world_size;
        int start = world_rank * chunk_size;
        int end = (world_rank == world_size - 1) ? n : start + chunk_size;
        int local_size = end - start;

        for (int i = 0; i < n - 1; ++i) {
            bool local_updated = false;
            vector<int> local_distances = result.distances;
            vector<int> local_predecessors = result.predecessors;

            #pragma omp parallel for reduction(||:local_updated)
            for (int u = 0; u < n; ++u) {
                if (result.distances[u] == INT_MAX) continue;

                for (int v = start; v < end; ++v) {
                    int weight = get(u, v);
                    if (weight != 0 && result.distances[u] != INT_MAX) {
                        long long new_dist = (long long)result.distances[u] + weight;
                        if (new_dist < INT_MAX && new_dist < result.distances[v]) {
                            #pragma omp critical
                            {
                                local_distances[v] = new_dist;
                                local_predecessors[v] = u;
                                local_updated = true;
                            }
                        }
                    }
                }
            }

            vector<int> all_distances(n);
            MPI_Allreduce(local_distances.data(), all_distances.data(), n, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

            for (int v = 0; v < n; ++v) {
                if (all_distances[v] != result.distances[v]) {
                    result.distances[v] = all_distances[v];
                    result.predecessors[v] = local_predecessors[v];
                }
            }

            int global_updated;
            MPI_Allreduce(&local_updated, &global_updated, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

            result.iterations++;

            if (!global_updated) break;
        }

        bool local_negative_cycle = false;

        #pragma omp parallel for reduction(||:local_negative_cycle)
        for (int u = 0; u < n; ++u) {
            if (result.distances[u] == INT_MAX) continue;

            for (int v = start; v < end; ++v) {
                int weight = get(u, v);
                if (weight != 0 && result.distances[u] != INT_MAX) {
                    long long new_dist = (long long)result.distances[u] + weight;
                    if (new_dist < result.distances[v]) {
                        local_negative_cycle = true;
                        #pragma omp critical
                        {
                            result.distances[v] = INT_MIN;
                        }
                    }
                }
            }
        }

        int global_negative_cycle;
        MPI_Allreduce(&local_negative_cycle, &global_negative_cycle, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        result.hasNegativeCycle = global_negative_cycle;

        MPI_Allreduce(MPI_IN_PLACE, result.distances.data(), n, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        return result;
    }
};

class UndirectedGraph : public Graph {
public:
    UndirectedGraph(int vertices) : Graph(vertices) {}
    void generateGraph(bool allowNegativeWeights) override {
        std::random_device rd;
        std::mt19937 gen(rd());
        int minW = allowNegativeWeights ? -10 : 1;
        std::uniform_int_distribution<int> dist(minW, 10);

        #pragma omp parallel for
        for (int i = 0; i < getRows(); ++i) {
            for (int j = i + 1; j < getCols(); ++j) {
                if (dist(gen) > 5) {
                    int weight = dist(gen);
                    #pragma omp critical
                    {
                        set(i, j, weight);
                        set(j, i, weight);
                    }
                }
            }
        }
    }

    void generateGraphDistributed(bool allowNegativeWeights, int world_rank, int world_size) {
        std::random_device rd;
        std::mt19937 gen(rd() + world_rank);

        int n = getRows();
        int chunk_size = n / world_size;
        int remainder = n % world_size;
        int start = world_rank * chunk_size + std::min(world_rank, remainder);
        int end = start + chunk_size + (world_rank < remainder ? 1 : 0);

        for (int i = start; i < end; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double prob = static_cast<double>(gen() % 100) / 100.0;
                if (prob > 0.3) {
                    int weight;
                    do {
                        weight = (allowNegativeWeights ? (gen() % 21 - 10) : (gen() % 10 + 1));
                    } while (weight == 0);

                    set(i, j, weight);
                    set(j, i, weight);
                }
            }
        }

        syncMatrixUndirected(world_rank, world_size);
    }


    void printInfo() const override {
        std::cout << "Undirected Graph with " << getRows() << " vertices\n";
        printWeightMatrix();
    }

    void addEdge(int from, int to, int weight) {
        set(from, to, weight);
        set(to, from, weight);
    }

    struct Edge {
        int u, v, weight;
        Edge() : u(0), v(0), weight(0) {}
        Edge(int u, int v, int w) : u(u), v(v), weight(w) {}
        bool operator<(const Edge& other) const {
            return weight < other.weight;
        }
    };

    struct MSTResult {
        std::vector<Edge> edges;
        int totalWeight = 0;
        bool success = false;
        std::string message;
    };

    class DSU {
    public:
        std::vector<int> parent, rank;

        DSU(int n) {
            parent.resize(n);
            rank.resize(n, 0);
            #pragma omp parallel for
            for (int i = 0; i < n; i++) parent[i] = i;
        }

        int find(int x) {
            if (parent[x] != x) {
                parent[x] = find(parent[x]);
            }
            return parent[x];
        }

        void unite(int x, int y) {
            int rootX = find(x);
            int rootY = find(y);

            if (rootX != rootY) {
                if (rank[rootX] < rank[rootY]) {
                    parent[rootX] = rootY;
                } else if (rank[rootX] > rank[rootY]) {
                    parent[rootY] = rootX;
                } else {
                    parent[rootY] = rootX;
                    rank[rootX]++;
                }
            }
        }
    };


    MSTResult kruskalMST() const {
        MSTResult result;
        int n = getRows();

        std::vector<Edge> edges;
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int weight = get(i, j);
                if (weight != 0) {
                    #pragma omp critical
                    edges.push_back(Edge(i, j, weight));
                }
            }
        }

        #pragma omp parallel
        {
            #pragma omp single
            std::sort(edges.begin(), edges.end());
        }

        DSU dsu(n);
        result.totalWeight = 0;
        int edgesUsed = 0;

        int local_totalWeight = 0;
        int local_edgesUsed = 0;
        #pragma omp parallel for reduction(+:local_totalWeight, local_edgesUsed)
        for (size_t i = 0; i < edges.size(); i++) {
            const Edge& edge = edges[i];
            if (dsu.find(edge.u) != dsu.find(edge.v)) {
                #pragma omp critical
                {
                    if (dsu.find(edge.u) != dsu.find(edge.v)) {
                        result.edges.push_back(edge);
                        local_totalWeight += edge.weight;
                        dsu.unite(edge.u, edge.v);
                        local_edgesUsed++;
                    }
                }
            }
        }
        result.totalWeight = local_totalWeight;
        edgesUsed = local_edgesUsed;

        if (edgesUsed == n - 1) {
            result.success = true;
            result.message = "Минимальное остовное дерево построено";
        } else {
            result.success = false;
            result.message = "Граф несвязный, невозможно построить MST";
        }

        return result;
    }

    MSTResult kruskalMSTDistributed(int world_rank, int world_size) const {
        MSTResult result;
        int n = getRows();

        std::vector<Edge> local_edges;
        for (int i = world_rank; i < n; i += world_size) {
            for (int j = i + 1; j < n; j++) {
                int weight = get(i, j);
                if (weight != 0) {
                    local_edges.push_back(Edge(i, j, weight));
                }
            }
        }

        if (world_rank == ROOT_RANK) {
            std::vector<Edge> all_edges = local_edges;

            for (int i = 1; i < world_size; i++) {
                int edge_count;
                MPI_Recv(&edge_count, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                std::vector<Edge> temp_edges(edge_count);
                MPI_Recv(temp_edges.data(), edge_count * 3, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                all_edges.insert(all_edges.end(), temp_edges.begin(), temp_edges.end());
            }

            std::sort(all_edges.begin(), all_edges.end());
            DSU dsu(n);
            result.totalWeight = 0;
            int edgesUsed = 0;

            for (const Edge& edge : all_edges) {
                if (dsu.find(edge.u) != dsu.find(edge.v)) {
                    result.edges.push_back(edge);
                    result.totalWeight += edge.weight;
                    dsu.unite(edge.u, edge.v);
                    edgesUsed++;

                    if (edgesUsed == n - 1) break;
                }
            }

            result.success = (edgesUsed == n - 1);
            result.message = result.success ? "MST построено распределенно" : "Граф несвязный";

        } else {
            int edge_count = local_edges.size();
            MPI_Send(&edge_count, 1, MPI_INT, ROOT_RANK, 0, MPI_COMM_WORLD);

            std::vector<int> edge_data;
            for (const Edge& edge : local_edges) {
                edge_data.push_back(edge.u);
                edge_data.push_back(edge.v);
                edge_data.push_back(edge.weight);
            }

            MPI_Send(edge_data.data(), edge_count * 3, MPI_INT, ROOT_RANK, 1, MPI_COMM_WORLD);
        }

        return result;
    }

    struct ColoringResult {
        std::vector<int> colors;
        int numberOfColors = 0;
        bool success = false;
        std::string message;
    };


    ColoringResult greedyColoring() const {
        ColoringResult result;
        int n = getRows();
        result.colors.resize(n, -1);

        if (n == 0) {
            result.message = "Граф пуст";
            return result;
        }

        result.colors[0] = 0;

        #pragma omp parallel for
        for (int u = 1; u < n; u++) {
            std::vector<bool> available(n, true);

            for (int v = 0; v < n; v++) {
                if (get(u, v) != 0 && result.colors[v] != -1) {
                    available[result.colors[v]] = false;
                }
            }

            int color;
            for (color = 0; color < n; color++) {
                if (available[color]) break;
            }

            result.colors[u] = color;
        }

        int maxColor = 0;
        #pragma omp parallel for reduction(max:maxColor)
        for (int color : result.colors) {
            if (color > maxColor) maxColor = color;
        }
        result.numberOfColors = maxColor + 1;
        result.success = true;
        result.message = "Раскраска завершена";

        return result;
    }

    ColoringResult greedyColoringDistributed(int world_rank, int world_size) const {
        ColoringResult result;
        int n = getRows();
        result.colors.resize(n, -1);

        if (n == 0) {
            result.message = "Граф пуст";
            return result;
        }

        int chunk_size = n / world_size;
        int remainder = n % world_size;
        int start = world_rank * chunk_size;
        int end = (world_rank == world_size - 1) ? n : start + chunk_size;
        int local_size = end - start;

        if (world_rank == 0) {
            result.colors[0] = 0;
        }

        MPI_Bcast(result.colors.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

        bool changed = true;
        int iterations = 0;
        const int MAX_ITERATIONS = n * 2;

        while (changed && iterations < MAX_ITERATIONS) {
            changed = false;
            bool local_changed = false;

            #pragma omp parallel for reduction(||:local_changed)
            for (int local_idx = 0; local_idx < local_size; local_idx++) {
                int u = start + local_idx;

                if (result.colors[u] != -1) continue;

                std::vector<bool> available(n, true);

                for (int v = 0; v < n; v++) {
                    if (get(u, v) != 0 && result.colors[v] != -1) {
                        available[result.colors[v]] = false;
                    }
                }

                int color = -1;
                for (int c = 0; c < n; c++) {
                    if (available[c]) {
                        color = c;
                        break;
                    }
                }

                if (color != -1 && color != result.colors[u]) {
                    result.colors[u] = color;
                    local_changed = true;
                }
            }

            MPI_Allreduce(MPI_IN_PLACE, result.colors.data(), n, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

            MPI_Allreduce(&local_changed, &changed, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
            iterations++;
        }

        vector<int> local_color_presence(n, 0);
        vector<int> global_color_presence(n, 0);

        #pragma omp parallel for
        for (int local_idx = 0; local_idx < local_size; local_idx++) {
            int u = start + local_idx;
            if (result.colors[u] != -1) {
                local_color_presence[result.colors[u]] = 1;
            }
        }

        MPI_Reduce(local_color_presence.data(), global_color_presence.data(), n,
                   MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            int maxColor = -1;
            int colorCount = 0;

            for (int i = 0; i < n; i++) {
                if (global_color_presence[i] == 1) {
                    colorCount++;
                    maxColor = i;
                }
            }

            result.numberOfColors = colorCount;
            result.success = (colorCount > 0);
            result.message = result.success ?
                "Распределенная раскраска завершена" :
                "Не удалось раскрасить граф";
        }

        MPI_Bcast(&result.success, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        MPI_Bcast(&result.numberOfColors, 1, MPI_INT, 0, MPI_COMM_WORLD);

        return result;
    }

    struct EulerStatus {
        enum Type { EULERIAN, SEMI_EULERIAN, NOT_CONNECTED, NEITHER };
        Type type = NEITHER;
        std::string message;
        int oddDegreeCount = 0;
        std::vector<int> oddDegreeVertices;
    };

    EulerStatus checkEulerianProperties() const {
        EulerStatus status;
        int n = getRows();

        if (n == 0) {
            status.type = EulerStatus::EULERIAN;
            status.message = "Граф пуст";
            return status;
        }


        if (!isConnected()) {
            status.type = EulerStatus::NOT_CONNECTED;
            status.message = "Граф несвязный";
            return status;
        }

        std::vector<int> degrees(n, 0);
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (get(i, j) != 0) {
                    #pragma omp atomic
                    degrees[i]++;
                }
            }
        }

        int local_oddCount = 0;
        #pragma omp parallel for reduction(+:local_oddCount)
        for (int i = 0; i < n; i++) {
            if (degrees[i] % 2 != 0) {
                local_oddCount++;
                #pragma omp critical
                status.oddDegreeVertices.push_back(i);
            }
        }
        status.oddDegreeCount = local_oddCount;

        if (status.oddDegreeCount == 0) {
            status.type = EulerStatus::EULERIAN;
            status.message = "Граф эйлеров";
        } else if (status.oddDegreeCount == 2) {
            status.type = EulerStatus::SEMI_EULERIAN;
            status.message = "Граф полуэйлеров";
        } else {
            status.type = EulerStatus::NEITHER;
            status.message = "Граф не эйлеров";
        }

        return status;
    }

private:
    bool isConnected() const {
        int n = getRows();
        if (n <= 1) return true;

        std::vector<bool> visited(n, 0);
        std::queue<int> q;
        q.push(0);
        visited[0] = 1;
        int visitedCount = 1;

        while (!q.empty()) {
            int u = q.front();
            q.pop();

            #pragma omp parallel for
            for (int v = 0; v < n; v++) {
                if (get(u, v) != 0 && visited[v] == 0) {
                    #pragma omp critical
                    {
                        if (visited[v] == 0) {
                            visited[v] = 1;
                            q.push(v);
                            visitedCount++;
                        }
                    }
                }
            }
        }

        return visitedCount == n;
    }

    void syncMatrixUndirected(int world_rank, int world_size) {
        int n = getRows();
        std::vector<int> flatMatrix(n * n);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                flatMatrix[i * n + j] = get(i, j);
            }
        }

        MPI_Allreduce(MPI_IN_PLACE, flatMatrix.data(), n * n, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                set(i, j, flatMatrix[i * n + j]);
            }
        }
    }
};


vector<int> bfsTraversal(const DirectedGraph& graph, int start) {
    int n = graph.getRows();
    vector<bool> visited(n, false);
    vector<int> result;
    queue<int> q;

    visited[start] = true;
    q.push(start);

    while (!q.empty()) {
        int current = q.front();
        q.pop();
        result.push_back(current);

        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            if (graph.get(current, i) != 0 && !visited[i]) {
                #pragma omp critical
                {
                    if (!visited[i]) {
                        visited[i] = true;
                        q.push(i);
                    }
                }
            }
        }
    }

    return result;
}

vector<int> bfsTraversalDistributed(const DirectedGraph& graph, int start, int world_rank, int world_size) {
    int n = graph.getRows();
    vector<int> visited(n, 0);
    vector<int> result;

    vector<int> current_frontier;
    vector<int> next_frontier;

    if (world_rank == 0) {
        visited[start] = 1;
        current_frontier.push_back(start);
        result.push_back(start);
    }

    MPI_Bcast(visited.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

    while (true) {
        int frontier_size = current_frontier.size();
        MPI_Bcast(&frontier_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (frontier_size == 0) break;

        current_frontier.resize(frontier_size);
        if (world_rank != 0) {
            current_frontier.resize(frontier_size);
        }
        MPI_Bcast(current_frontier.data(), frontier_size, MPI_INT, 0, MPI_COMM_WORLD);

        int local_chunk_size = frontier_size / world_size;
        int local_remainder = frontier_size % world_size;
        int local_start = world_rank * local_chunk_size;
        int local_end = local_start + local_chunk_size;
        if (world_rank == world_size - 1) {
            local_end += local_remainder;
        }

        vector<int> local_next_frontier;

        #pragma omp parallel
        {
            vector<int> thread_next_frontier;

            #pragma omp for nowait
            for (int idx = local_start; idx < local_end; idx++) {
                int current = current_frontier[idx];

                for (int neighbor = 0; neighbor < n; neighbor++) {
                    if (graph.get(current, neighbor) != 0 && visited[neighbor] == 0) {
                        thread_next_frontier.push_back(neighbor);
                    }
                }
            }

            #pragma omp critical
            {
                local_next_frontier.insert(local_next_frontier.end(),
                                         thread_next_frontier.begin(), thread_next_frontier.end());
            }
        }

        sort(local_next_frontier.begin(), local_next_frontier.end());
        local_next_frontier.erase(unique(local_next_frontier.begin(), local_next_frontier.end()),
                                local_next_frontier.end());

        vector<int> all_candidates;
        vector<int> recv_counts(world_size);
        vector<int> displs(world_size);

        int local_size = local_next_frontier.size();
        MPI_Gather(&local_size, 1, MPI_INT,
                  recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < world_size; i++) {
                displs[i] = displs[i-1] + recv_counts[i-1];
            }
            int total_candidates = displs[world_size-1] + recv_counts[world_size-1];
            all_candidates.resize(total_candidates);
        }

        MPI_Gatherv(local_next_frontier.data(), local_size, MPI_INT,
                   all_candidates.data(), recv_counts.data(), displs.data(), MPI_INT,
                   0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            sort(all_candidates.begin(), all_candidates.end());
            all_candidates.erase(unique(all_candidates.begin(), all_candidates.end()),
                               all_candidates.end());

            for (int candidate : all_candidates) {
                if (visited[candidate] == 0) {
                    visited[candidate] = 1;
                    next_frontier.push_back(candidate);
                    result.push_back(candidate);
                }
            }

            current_frontier = std::move(next_frontier);
            next_frontier.clear();
        }

        MPI_Bcast(visited.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

        int next_frontier_size = current_frontier.size();
        MPI_Bcast(&next_frontier_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (next_frontier_size == 0) break;
    }

    return result;
}

vector<UndirectedGraph::Edge> boruvkaMST(const UndirectedGraph& graph) {
    cout << "\n=== Алгоритм Борувки ===" << endl;

    int n = graph.getRows();
    vector<UndirectedGraph::Edge> edges;
    vector<UndirectedGraph::Edge> mst;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            int weight = graph.get(i, j);
            if (weight != 0) {
                #pragma omp critical
                edges.push_back(UndirectedGraph::Edge(i, j, weight));
            }
        }
    }

    if (edges.empty()) {
        cout << "В графе нет ребер\n";
        return mst;
    }

    vector<int> component(n);
    #pragma omp parallel for
    for (int i = 0; i < n; i++) component[i] = i;

    int components = n;

    while (components > 1) {
        vector<int> minEdge(n, -1);

        #pragma omp parallel for
        for (int i = 0; i < edges.size(); i++) {
            int u = edges[i].u;
            int v = edges[i].v;
            int compU = component[u];
            int compV = component[v];

            if (compU == compV) continue;

            #pragma omp critical
            {
                if (minEdge[compU] == -1 || edges[minEdge[compU]].weight > edges[i].weight) {
                    minEdge[compU] = i;
                }
                if (minEdge[compV] == -1 || edges[minEdge[compV]].weight > edges[i].weight) {
                    minEdge[compV] = i;
                }
            }
        }

        for (int i = 0; i < n; i++) {
            if (minEdge[i] != -1) {
                int idx = minEdge[i];
                int u = edges[idx].u;
                int v = edges[idx].v;

                if (component[u] != component[v]) {
                    mst.push_back(edges[idx]);

                    int oldComp = component[v];
                    int newComp = component[u];
                    #pragma omp parallel for
                    for (int j = 0; j < n; j++) {
                        if (component[j] == oldComp) {
                            component[j] = newComp;
                        }
                    }
                    components--;
                }
            }
        }
    }

    cout << "Минимальное остовное дерево содержит " << mst.size() << " ребер:\n";
    int totalWeight = 0;
    for (const auto& e : mst) {
        cout << e.u << " - " << e.v << " (вес: " << e.weight << ")\n";
        totalWeight += e.weight;
    }
    cout << "Общий вес: " << totalWeight << endl;

    return mst;
}

vector<UndirectedGraph::Edge> boruvkaMSTDistributed(const UndirectedGraph& graph,
                                                   int world_rank, int world_size) {
    cout << "\n=== Алгоритм Борувки ===" << endl;
    int n = graph.getRows();
    vector<UndirectedGraph::Edge> mst;

    if (world_rank == ROOT_RANK) {
        vector<UndirectedGraph::Edge> all_edges;

        for (int proc = 0; proc < world_size; proc++) {
            if (proc == ROOT_RANK) {
                for (int i = 0; i < n; i++) {
                    for (int j = i + 1; j < n; j++) {
                        int weight = graph.get(i, j);
                        if (weight != 0) {
                            all_edges.push_back(UndirectedGraph::Edge(i, j, weight));
                        }
                    }
                }
            } else {
                int edge_count;
                MPI_Recv(&edge_count, 1, MPI_INT, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                vector<int> edge_data(edge_count * 3);
                MPI_Recv(edge_data.data(), edge_count * 3, MPI_INT, proc, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                for (int i = 0; i < edge_count; i++) {
                    all_edges.push_back(UndirectedGraph::Edge(
                        edge_data[i * 3],
                        edge_data[i * 3 + 1],
                        edge_data[i * 3 + 2]
                    ));
                }
            }
        }

        vector<int> component(n);
        for (int i = 0; i < n; i++) component[i] = i;
        int components = n;

        while (components > 1) {
            vector<int> minEdge(n, -1);

            for (int i = 0; i < all_edges.size(); i++) {
                int u = all_edges[i].u;
                int v = all_edges[i].v;
                int compU = component[u];
                int compV = component[v];

                if (compU == compV) continue;

                if (minEdge[compU] == -1 || all_edges[minEdge[compU]].weight > all_edges[i].weight) {
                    minEdge[compU] = i;
                }
                if (minEdge[compV] == -1 || all_edges[minEdge[compV]].weight > all_edges[i].weight) {
                    minEdge[compV] = i;
                }
            }

            for (int i = 0; i < n; i++) {
                if (minEdge[i] != -1) {
                    int idx = minEdge[i];
                    int u = all_edges[idx].u;
                    int v = all_edges[idx].v;

                    if (component[u] != component[v]) {
                        mst.push_back(all_edges[idx]);

                        int oldComp = component[v];
                        int newComp = component[u];
                        for (int j = 0; j < n; j++) {
                            if (component[j] == oldComp) {
                                component[j] = newComp;
                            }
                        }
                        components--;
                    }
                }
            }
        }

    } else {
        vector<UndirectedGraph::Edge> local_edges;
        for (int i = world_rank; i < n; i += world_size) {
            for (int j = i + 1; j < n; j++) {
                int weight = graph.get(i, j);
                if (weight != 0) {
                    local_edges.push_back(UndirectedGraph::Edge(i, j, weight));
                }
            }
        }

        int edge_count = local_edges.size();
        MPI_Send(&edge_count, 1, MPI_INT, ROOT_RANK, 0, MPI_COMM_WORLD);

        vector<int> edge_data;
        for (const auto& edge : local_edges) {
            edge_data.push_back(edge.u);
            edge_data.push_back(edge.v);
            edge_data.push_back(edge.weight);
        }

        MPI_Send(edge_data.data(), edge_count * 3, MPI_INT, ROOT_RANK, 1, MPI_COMM_WORLD);
    }

    return mst;
}

void fanoCodingDemo() {
    cout << "\n=== Алгоритм Фано ===" << endl;

    vector<pair<char, double>> symbols = {
        {'A', 0.4},
        {'B', 0.2},
        {'C', 0.2},
        {'D', 0.1},
        {'E', 0.1}
    };

    sort(symbols.begin(), symbols.end(),
         [](const pair<char, double>& a, const pair<char, double>& b) {
             return a.second > b.second;
         });

    cout << "Коды Фано:\n";

    function<void(int, int, string)> buildCodes = [&](int start, int end, string code) {
        if (start == end) {
            cout << "Символ " << symbols[start].first << ": " << code << endl;
            return;
        }

        double total = 0;
        for (int i = start; i <= end; i++) {
            total += symbols[i].second;
        }

        double half = total / 2;
        double sum = 0;
        int split = start;

        for (int i = start; i <= end; i++) {
            sum += symbols[i].second;
            if (sum >= half) {
                split = i;
                break;
            }
        }

        buildCodes(start, split, code + "0");
        buildCodes(split + 1, end, code + "1");
    };

    buildCodes(0, symbols.size() - 1, "");
}

void bellmanFordDemo(const DirectedGraph& graph, int source) {
    cout << "\n=== Алгоритм Беллмана-Форда ===" << endl;

    auto result = graph.bellmanFord(source);

    cout << "Расстояния от вершины " << source << ":\n";
    for (int i = 0; i < result.distances.size(); i++) {
        if (result.distances[i] == INT_MAX) {
            cout << "Вершина " << i << ": недостижима\n";
        } else if (result.distances[i] == INT_MIN) {
            cout << "Вершина " << i << ": -INF (отрицательный цикл)\n";
        } else {
            cout << "Вершина " << i << ": " << result.distances[i] << "\n";
        }
    }

    if (result.hasNegativeCycle) {
        cout << "Обнаружен отрицательный цикл!\n";
    }
}

void initializeParallelEnvironment(int& world_rank, int& world_size) {
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const char* omp_env = getenv("OMP_NUM_THREADS");
    if (omp_env == nullptr) {
        omp_set_num_threads(MAX_OMP_THREADS);
    } else {
        int requested_threads = std::stoi(omp_env);
        if (requested_threads > MAX_OMP_THREADS) {
            if (world_rank == ROOT_RANK) {
                cout << "ПРЕДУПРЕЖДЕНИЕ: Запрошено " << requested_threads
                     << " потоков. Ограничено до " << MAX_OMP_THREADS << endl;
            }
            omp_set_num_threads(MAX_OMP_THREADS);
        }
    }

    if (world_rank == ROOT_RANK) {
        cout << "Параллельное окружение инициализировано\n";
        cout << "MPI процессов: " << world_size << endl;
        cout << "OpenMP потоков на процесс: " << omp_get_max_threads() << endl;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    initializeParallelEnvironment(world_rank, world_size);

    if (world_rank == ROOT_RANK) {
        cout << "=== РАСПРЕДЕЛЕННАЯ ДЕМОНСТРАЦИЯ АЛГОРИТМОВ НА ГРАФАХ ===" << endl;
        cout << "Количество MPI процессов: " << world_size << endl;
        cout << "Максимальное количество потоков OpenMP: " << omp_get_max_threads() << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == ROOT_RANK) {
        cout << "\n--- ОРИЕНТИРОВАННЫЙ ГРАФ ---" << endl;
    }

    int graph_size = 6;
    if (world_rank == ROOT_RANK) {
        cout << "Введите количество вершин в графе: ";
        cin >> graph_size;
    }

    MPI_Bcast(&graph_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    DirectedGraph dirGraph(graph_size);
    dirGraph.generateGraphDistributed(true, world_rank, world_size);

    if (world_rank == ROOT_RANK) {
        dirGraph.printInfo();
    }

    if (world_rank == ROOT_RANK) {
        cout << "\n=== РАСПРЕДЕЛЕННЫЙ ОБХОД В ШИРИНУ (BFS) ===" << endl;
    }

    auto start_bfs = MPI_Wtime();
    vector<int> bfsOrder = bfsTraversalDistributed(dirGraph, 0, world_rank, world_size);
    auto end_bfs = MPI_Wtime();

    if (world_rank == ROOT_RANK) {
        cout << "Порядок обхода BFS: ";
        for (int v : bfsOrder) {
            cout << v << " ";
        }
        cout << endl;
        cout << "Время выполнения BFS: " << (end_bfs - start_bfs) * 1000 << " мс" << endl;
    }

    if (world_rank == ROOT_RANK) {
        cout << "\n=== РАСПРЕДЕЛЕННЫЙ АЛГОРИТМ БЕЛЛМАНА-ФОРДА ===" << endl;
    }

    auto start_bf = MPI_Wtime();
    auto bf_result = dirGraph.bellmanFordDistributed(0, world_rank, world_size);
    auto end_bf = MPI_Wtime();

    if (world_rank == ROOT_RANK) {
        cout << "Расстояния от вершины 0:\n";
        for (int i = 0; i < bf_result.distances.size(); i++) {
            if (bf_result.distances[i] == INT_MAX) {
                cout << "Вершина " << i << ": недостижима\n";
            } else if (bf_result.distances[i] == INT_MIN) {
                cout << "Вершина " << i << ": -INF (отрицательный цикл)\n";
            } else {
                cout << "Вершина " << i << ": " << bf_result.distances[i] << "\n";
            }
        }

        if (bf_result.hasNegativeCycle) {
            cout << "Обнаружен отрицательный цикл!\n";
        }
        cout << "Количество итераций: " << bf_result.iterations << endl;
        cout << "Время выполнения Беллмана-Форда: " << (end_bf - start_bf) * 1000 << " мс" << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == ROOT_RANK) {
        cout << "\n--- НЕОРИЕНТИРОВАННЫЙ ГРАФ ---" << endl;
    }

    UndirectedGraph undirGraph(6);
    undirGraph.generateGraphDistributed(false, world_rank, world_size);

    if (world_rank == ROOT_RANK) {
        undirGraph.printInfo();
    }

    if (world_rank == ROOT_RANK) {
        cout << "\n=== РАСПРЕДЕЛЕННЫЙ АЛГОРИТМ КРУСКАЛА ===" << endl;
    }

    auto start_kruskal = MPI_Wtime();
    auto kruskalResult = undirGraph.kruskalMSTDistributed(world_rank, world_size);
    auto end_kruskal = MPI_Wtime();

    if (world_rank == ROOT_RANK) {
        if (kruskalResult.success) {
            cout << "MST построено. Общий вес: " << kruskalResult.totalWeight << endl;
            cout << "Ребра MST (" << kruskalResult.edges.size() << " ребер):\n";
            for (const auto& edge : kruskalResult.edges) {
                cout << edge.u << " - " << edge.v << " (вес: " << edge.weight << ")\n";
            }
        } else {
            cout << "Ошибка: " << kruskalResult.message << endl;
        }
        cout << "Время выполнения Крускала: " << (end_kruskal - start_kruskal) * 1000 << " мс" << endl;
    }

    if (world_rank == ROOT_RANK) {
        cout << "\n=== РАСПРЕДЕЛЕННЫЙ АЛГОРИТМ БОРУВКИ ===" << endl;
    }

    auto start_boruvka = MPI_Wtime();
    auto boruvkaResult = boruvkaMSTDistributed(undirGraph, world_rank, world_size);
    auto end_boruvka = MPI_Wtime();

    if (world_rank == ROOT_RANK) {
        cout << "Алгоритм Борувки завершен. Найдено " << boruvkaResult.size() << " ребер MST" << endl;
        int totalWeight = 0;
        for (const auto& edge : boruvkaResult) {
            totalWeight += edge.weight;
        }
        cout << "Общий вес MST (Борувка): " << totalWeight << endl;
        cout << "Время выполнения Борувки: " << (end_boruvka - start_boruvka) * 1000 << " мс" << endl;
    }

    if (world_rank == ROOT_RANK) {
        cout << "\n=== РАСПРЕДЕЛЕННАЯ РАСКРАСКА ГРАФА ===" << endl;
    }

    auto start_coloring = MPI_Wtime();
    auto coloringResult = undirGraph.greedyColoringDistributed(world_rank, world_size);
    auto end_coloring = MPI_Wtime();

    if (world_rank == ROOT_RANK) {
        if (coloringResult.success) {
            cout << "Использовано цветов: " << coloringResult.numberOfColors << endl;
            cout << "Раскраска вершин:\n";
            for (int i = 0; i < coloringResult.colors.size(); i++) {
                cout << "Вершина " << i << ": цвет " << coloringResult.colors[i] << endl;
            }
            cout << "Качество раскраски: " << coloringResult.numberOfColors << " цветов" << endl;
        } else {
            cout << "Ошибка раскраски: " << coloringResult.message << endl;
        }
        cout << "Время выполнения раскраски: " << (end_coloring - start_coloring) * 1000 << " мс" << endl;
    }

    if (world_rank == ROOT_RANK) {
        cout << "\n=== ЭЙЛЕРОВЫ СВОЙСТВА ===" << endl;
        auto start_euler = MPI_Wtime();
        auto eulerStatus = undirGraph.checkEulerianProperties();
        auto end_euler = MPI_Wtime();

        cout << "Результат: " << eulerStatus.message << endl;
        cout << "Вершин с нечетной степенью: " << eulerStatus.oddDegreeCount << endl;
        if (!eulerStatus.oddDegreeVertices.empty()) {
            cout << "Вершины с нечетной степенью: ";
            for (int v : eulerStatus.oddDegreeVertices) {
                cout << v << " ";
            }
            cout << endl;
        }
        cout << "Время проверки эйлеровых свойств: " << (end_euler - start_euler) * 1000 << " мс" << endl;
    }

    if (world_rank == ROOT_RANK) {
        cout << "\n=== АЛГОРИТМ ФАНО ===" << endl;
        auto start_fano = MPI_Wtime();
        fanoCodingDemo();
        auto end_fano = MPI_Wtime();
        cout << "Время выполнения алгоритма Фано: " << (end_fano - start_fano) * 1000 << " мс" << endl;
    }

    if (world_rank == ROOT_RANK) {
        cout << "\n=== СВОДКА ПРОИЗВОДИТЕЛЬНОСТИ ===" << endl;
        cout << "Все распределенные алгоритмы успешно выполнены!" << endl;
        cout << "Использовано процессов MPI: " << world_size << endl;
        cout << "Использовано потоков OpenMP на процесс: " << omp_get_max_threads() << endl;
        cout << "Размер графов: " << graph_size << " и 6 вершин" << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == ROOT_RANK) {
        cout << "\n=== ВСЕ РАСПРЕДЕЛЕННЫЕ АЛГОРИТМЫ ЗАВЕРШЕНЫ ===" << endl;
    }

    MPI_Finalize();
    return 0;
}