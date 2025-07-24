#include <fstream>
#include <iostream>
#include <unordered_set>
#include <vector>
#include <limits>
#include <queue>

using namespace std;

// 根據Random_Graph.py設定的vertex數調整
#define VERTEX_NUM 5
#define INF 1e9

unordered_set<int> terminals, vertex;
vector<vector<double>> dist(VERTEX_NUM, vector<double>(VERTEX_NUM, INF));

struct Edge {
    int u, v;
    double cost;

    Edge(int x, int y, double z) : u(x), v(y), cost(z) {}

    bool operator==(const Edge& other) const {
        return u == other.u && v == other.v;
    }
};

struct EdgeHash {
    size_t operator()(const Edge& e) const {
        return hash<int>()(e.u) ^ (hash<int>()(e.v) << 1);
    }
};

struct Tree {
    int root;
    vector<Edge> edges; 
    double cost = 0.0; 
    unordered_set<int> terms;
    double density() const {
        return (terms.empty() || edges.size() == 0) ? INF : cost / double(terms.size());
    }
    // bool empty() const { return edges.empty() && nodes.empty(); }
    Tree(int r) {
        root = r;
        Insert(r);
    }

    Tree() {
        root = -1;
    }

    void Insert(int v) {
        if(terminals.find(v) != terminals.end())
        {
            terms.insert(v);
        }
    }
};

Tree Union_Tree(Tree T_RETURN, Tree T_BEST)
{
    unordered_set<Edge, EdgeHash> edge_set(T_RETURN.edges.begin(), T_RETURN.edges.end());

    for (const auto& e : T_BEST.edges) {
        if (edge_set.find(e) == edge_set.end()) {
            T_RETURN.edges.push_back(e);
            T_RETURN.cost += e.cost;
            edge_set.insert(e);
        }
    }

    T_RETURN.terms.insert(T_BEST.terms.begin(), T_BEST.terms.end());
    return T_RETURN;
}

int Intersection_Vertex(unordered_set<int> T, unordered_set<int>& V)
{
    int cnt = 0;
    for(auto it = T.begin(); it != T.end(); it++)
    {
        if(V.find(*it) != V.end())
        {
            V.erase(*it);
            cnt++;
        }
    }
    return cnt;
}

int Calculate_Reachable(int r, unordered_set<int>& V, int l)
{
    queue<int> q;
    q.push(r);
    int result = 0;
    unordered_set<int> records;
    records.insert(r);

    while(!q.empty() && l > 0)
    {
        int curr = q.front();
        q.pop();

        for(int i = 0; i < VERTEX_NUM; i++)
        {
            if(abs(dist[curr][i] - INF) > 0.01 && records.find(i) == records.end())
            {
                q.push(i);
                records.insert(i);
            }
        }

        l--;
    }

    for(auto it = records.begin(); it != records.end(); it++)
    {
        if(V.find(*it) != V.end())
        {
            result++;
        }
    }

    return result;
}

Tree D_Steiner(int level, int k, int r, unordered_set<int> V)
{
    Tree T_RETURN(r);
    int reachable = Calculate_Reachable(r, V, level);

    if(level == 0 || k > reachable)
    {
        return T_RETURN;
    }
    while(k > 0)
    {
        Tree T_BEST;
        for(int i = 0; i < VERTEX_NUM; i++)
        {
            // 剪枝避免複雜度炸開
            if (abs(dist[r][i] - INF) < 1e-6) continue; 
            for(int j = 1; j <= k; j++)
            {
                Tree T_TMP(r);
                if(r != i)
                {
                    T_TMP.edges.push_back(Edge(r, i, dist[r][i]));
                    T_TMP.cost += dist[r][i];
                    T_TMP.Insert(i);
                }

                T_TMP = Union_Tree(T_TMP, D_Steiner(level - 1, j, i, V));
                if(T_BEST.density() > T_TMP.density())
                {
                    T_BEST = T_TMP;
                }
            }
        }
        
        T_RETURN = Union_Tree(T_RETURN, T_BEST);
        int delta = Intersection_Vertex(T_BEST.terms, V);
        if (delta == 0) {
            cout << "Warning: Cannot find a tree covering all " << k << " terminals. Returning best partial tree." << endl;
            cout << "level:" << level << " r:" << r << endl;
            break;
        }
        k -= delta;
    }

    return T_RETURN;
}

int main() {
    ifstream fin("edges.txt");
    if (!fin) {
        cerr << "cannot open edges.txt\n";
        return 1;
    }
    int u, v, w;
    while (fin >> u >> v >> w) {
        if(u < 2)
        {
            terminals.insert(u);
        }

        if(v < 2)
        {
            terminals.insert(v);
        }

        vertex.insert(u);
        vertex.insert(v);
        dist[u][v] = w;
    }

    Tree T_BEST = D_Steiner(2, 2, 2, terminals);
    cout << "Density: " << T_BEST.density() << endl;
    return 0;
}
