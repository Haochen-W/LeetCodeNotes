#include <vector>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <string>
#include <iostream>
#include <sstream>
#include <memory>
#include <stack>
#include <set>
#include <boost/algorithm/string.hpp>

using namespace std;

namespace LinkedList
{
    struct ListNode
    {
        int val;
        ListNode *next;
        ListNode() : val(0), next(nullptr) {}
        ListNode(int x) : val(x), next(nullptr) {}
        ListNode(int x, ListNode *next) : val(x), next(next) {}
    };
    ListNode *middleNode(ListNode *head)
    {
        if (!head)
            return nullptr;
        ListNode *slow = head, *fast = head;
        while (fast && fast->next)
        {
            slow = slow->next;
            fast = fast->next->next;
        }
        // 1, 2, 3, 4, 5, 6    -> return 4
        // 1, 2, 3, 4, 5, 6, 7 -> return 4
        return slow;
    }
} // namespace LinkedList

namespace unionFind
{
    struct DisjointSetUnionBySize
    {
        vector<int> parent;
        vector<int> size;
        int groups;

        DisjointSetUnionBySize(int maxSize)
        {
            parent.resize(maxSize);
            size.resize(maxSize);
            for (int i = 0; i < maxSize; i++)
            {
                parent[i] = i;
                size[i] = 1;
            }
            groups = maxSize;
        }

        int find_set(int v)
        {
            if (v == parent[v])
                return v;
            return parent[v] = find_set(parent[v]);
        }

        int find_set_iter(int v)
        {
            while (v != parent[v])
            {
                parent[v] = parent[parent[v]];
                v = parent[v];
            }
            return v;
        }

        bool union_set(int a, int b)
        {
            a = find_set(a);
            b = find_set(b);
            if (a != b)
            {
                if (size[a] < size[b])
                    swap(a, b);
                parent[b] = a;
                size[a] += size[b];
                groups--;
                return true;
            }
            return false;
        }
        int find_groups()
        {
            return groups;
        }
    };

    struct DisjointSetUnionByRank
    {
        vector<int> parent;
        vector<int> size;
        vector<int> rank;
        int groups;

        DisjointSetUnionByRank(int maxSize)
        {
            parent.resize(maxSize);
            size.resize(maxSize);
            rank.resize(maxSize);
            groups = maxSize;
            for (int i = 0; i < maxSize; i++)
            {
                parent[i] = i;
                size[i] = 1;
                rank[i] = 0;
            }
        }

        int find_set(int v)
        {
            if (v == parent[v])
                return v;
            return parent[v] = find_set(parent[v]);
        }

        int find_set_iter(int v)
        {
            while (v != parent[v])
            {
                parent[v] = parent[parent[v]];
                v = parent[v];
            }
            return v;
        }

        bool union_set(int a, int b)
        {
            a = find_set(a);
            b = find_set(b);
            if (a != b)
            {
                if (rank[a] < rank[b])
                    swap(a, b);
                parent[b] = parent[a];
                size[a] += size[b];
                if (rank[a] == rank[b])
                    rank[a]++;
                return true;
            }
            return false;
        }
    };

    // for large number of nodes, e.g. sparse graph. Don't need to know the range of nodes in advance. We don't even need to know the type
    template <class T>
    class DisjointSetUnionBySizeT
    {
        unordered_map<T, T> parent;
        unordered_map<T, int> size;

    public:
        DisjointSetUnionBySizeT() {}

        const T &find(const T &v)
        {
            if (!parent.count(v))
            {
                parent[v] = v;
                size[v] = 1;
                return v;
            }
            if (v == parent[v])
                return v;
            return parent[v] = find(parent[v]);
        }

        const T &find_iter(const T &v)
        {
            if (!parent.count(v))
            {
                parent[v] = v;
                size[v] = 1;
                return v;
            }
            const T m = v;
            while (m != parent[m])
            {
                parent[m] = parent[parent[m]];
                m = parent[m];
            }
            return m;
        }

        bool add(const T &a, const T &b)
        {
            const T &aa = find(a);
            const T &bb = find(b);
            if (aa != bb)
            {
                if (size[aa] < size[bb])
                    swap(aa, bb);
                parent[bb] = aa;
                size[aa] += size[bb];
                return true;
            }
            return false;
        }
    };
} // namespace unionFind

namespace TarjanAlgo
{
    class CriticalConnection
    {
    private:
        vector<int> disc, low;
        int time = 1;
        vector<vector<int>> ans;
        vector<vector<int>> edgeMap;

    public:
        vector<vector<int>> criticalConnections(int n, vector<vector<int>> &connections)
        {
            disc.resize(n), low.resize(n);
            edgeMap.resize(n);
            for (auto conn : connections)
            {
                edgeMap[conn[0]].push_back(conn[1]);
                edgeMap[conn[1]].push_back(conn[0]);
            }
            dfs(0, -1);
            return ans;
        }
        void dfs(int curr, int prev)
        {
            disc[curr] = low[curr] = time++;
            for (int next : edgeMap[curr])
            {
                if (disc[next] == 0)
                {
                    dfs(next, curr);
                    low[curr] = min(low[curr], low[next]);
                }
                else if (next != prev)
                    low[curr] = min(low[curr], disc[next]);
                if (low[next] > disc[curr])
                    ans.push_back({curr, next});
            }
        }
    };
} // namespace TarjanAlgo

namespace binarySearch
{
    int bisect_left(vector<int> A, int x)
    {
        int l = 0, r = A.size() - 1;
        while (l <= r)
        {
            int mid = l + (r - l) / 2;
            if (A[mid] >= x)
                r = mid - 1;
            else
                l = mid + 1;
        }
        return l;
    }

    int bisect_right(vector<int> A, int x)
    {
        int l = 0, r = A.size() - 1;
        while (l <= r)
        {
            int mid = l + (r - l) / 2;
            if (A[mid] <= x)
                l = mid + 1;
            else
                r = mid - 1;
        }
        return l;
    }

    // minimize index k such that condition(k) is true, should be similar to bisect left
    int binary_search_mink(vector<int> A, std::function<bool(int mid)> condition)
    {
        int l = 0, r = A.size();
        while (l < r)
        {
            int mid = l + (r - l) / 2;
            if (condition(mid))
                r = mid;
            else
                l = mid + 1;
        }
        return l;
    }

    int binary_search_mink_v2(vector<int> A, std::function<bool(int mid)> condition)
    {
        int l = 0, r = A.size() - 1;
        while (l <= r)
        {
            int mid = l + (r - l) / 2;
            if (condition(mid))
                r = mid - 1;
            else
                l = mid + 1;
        }
        return l;
    }
} // namespace binarySearch

namespace TopologicalSort
{
    class DFSVersion
    {
        int n;                   // number of vertices
        vector<vector<int>> adj; // adjacency list of graph
        vector<bool> visited;
        vector<int> ans;

        void dfs(int v)
        {
            visited[v] = true;
            for (int u : adj[v])
            {
                if (!visited[u])
                    dfs(u);
            }
            ans.push_back(v);
        }

        void topological_sort()
        {
            visited.assign(n, false);
            ans.clear();
            for (int i = 0; i < n; ++i)
            {
                if (!visited[i])
                    dfs(i);
            }
            reverse(ans.begin(), ans.end());
        }
    };

    class DFSVersionIterative
    {
        vector<int> topological_sort(int n, vector<set<int>> graph, int start)
        {
            vector<int> visited(n, false);
            vector<int> stack, order;
            std::stack<int> s;
            s.push(start);
            while (s.size())
            {
                int curr = s.top();
                s.pop();
                if (!visited[curr])
                {
                    visited[curr] = true;
                    for (int neigh : graph[curr])
                    {
                        s.push(neigh);
                    }
                    while (stack.size() && !graph[stack.back()].count(curr))
                    {
                        order.push_back(stack.back());
                        stack.pop_back();
                    }
                }
            }
            // ans = stack;
            stack.insert(stack.end(), order.rbegin(), order.rend());
            return stack;
        }
    };

    class QueueVersion
    {
        bool canFinish(int n, vector<vector<int>> &edges)
        {
            int cnt = 0;
            vector<unordered_set<int>> m(n, unordered_set<int>());
            vector<int> par_cnt(n, 0);

            for (auto &v : edges)
            {
                m[v[0]].insert(v[1]);
                par_cnt[v[1]]++;
            }

            queue<int> q;
            for (int i = 0; i < n; i++)
            {
                if (par_cnt[i] == 0)
                    q.push(i);
            }
            cnt = q.size();

            while (q.size())
            {
                int curr = q.front();
                q.pop();
                for (int child : m[curr])
                {
                    if (--par_cnt[child] == 0)
                    {
                        cnt++;
                        q.push(child);
                    }
                }
            }
            return cnt == n;
        }
    };
} // namespace TopologicalSort

namespace Dijkstras
{
    void Dijkstras(int source, vector<vector<pair<int, int>>> &al, vector<long long> &visited, vector<int> &prev)
    {
        // @params:
        //   source : source node, the graph has node 0 ~ n-1. (assume n is known)
        //   al     : the graph. al[u] = {(v1, w1), (v2, w2), ...} edge u -> v1 with weight w1, etc.
        //   visited: when pass in, initialize to LLONG_MAX of size n
        // @result:
        //   visited: visited[i] is the min path length from source to node i.
        //   prev   : u = prev[i] means that in the min path from source to node i, the parent of node i is u.
        priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<pair<long long, int>>> pq;
        pq.push({0, source});
        prev[source] = -1;
        while (!pq.empty())
        {
            auto [dist, i] = pq.top();
            pq.pop();
            if (visited[i] != dist)
                continue;
            for (auto [j, w] : al[i])
            {
                if (visited[j] > dist + w)
                {
                    visited[j] = dist + w;
                    prev[j] = i;
                    pq.push({visited[j], j});
                }
            }
        }
    }
} // namespace Dijkstras

namespace Trie
{
    class Trie
    {
        class Node
        {
        public:
            map<char, Node *> child;
            bool isend = false;
        };
        Node head;

    public:
        Trie() {}

        void insert(string word)
        {
            Node *curr = &head;
            for (char c : word)
            {
                if (!curr->child.count(c))
                {
                    curr->child[c] = new Node;
                }
                curr = curr->child[c];
            }
            curr->isend = true;
        }

        bool search(string word)
        {
            Node *cnode = &head;
            for (int i = 0; i < word.size(); i++)
            {
                char c = word[i];
                if (cnode->child.count(c))
                {
                    cnode = cnode->child[c];
                }
                else
                {
                    return false;
                }
            }
            return cnode->isend;
        }

        bool startsWith(string prefix)
        {
            Node *cnode = &head;
            for (int i = 0; i < prefix.size(); i++)
            {
                char c = prefix[i];
                if (cnode->child.count(c))
                {
                    cnode = cnode->child[c];
                }
                else
                {
                    return false;
                }
            }
            return true;
        }
    };
} // namespace Trie

namespace MorrisTraversal
{
    struct TreeNode
    {
        int val;
        TreeNode *left;
        TreeNode *right;
    };
    vector<int> inorder(TreeNode *root)
    {
        vector<int> res;
        TreeNode *curr = root;
        while (curr)
        {
            if (!curr->left)
            {
                res.push_back(curr->val);
                curr = curr->right;
            }
            else
            {
                TreeNode *prev = curr->left;
                while (prev->right && prev->right != curr)
                    prev = prev->right;
                if (!prev->right)
                {
                    prev->right = curr;
                    curr = curr->left;
                }
                else
                {
                    // curr subtree traversed. revert the temp link and backtrack.
                    prev->right = nullptr;
                    res.push_back(curr->val);
                    curr = curr->right;
                }
            }
        }
        return res;
    }

    vector<int> preorder(TreeNode *root)
    {
        vector<int> res;
        TreeNode *curr = root;
        while (curr)
        {
            if (!curr->left)
            {
                res.push_back(curr->val);
                curr = curr->right;
            }
            else
            {
                TreeNode *prev = curr->left;
                while (prev->right && prev->right != curr)
                    prev = prev->right;
                if (!prev->right)
                {
                    res.push_back(curr->val);
                    prev->right = curr;
                    curr = curr->left;
                }
                else
                {
                    // curr subtree traversed. revert the temp link and backtrack.
                    prev->right = nullptr;
                    curr = curr->right;
                }
            }
        }
        return res;
    }

    vector<int> postorder(TreeNode *root)
    {
        vector<int> res;
        TreeNode *curr = root;
        while (curr)
        {
            if (!curr->right)
            {
                res.push_back(curr->val);
                curr = curr->left;
            }
            else
            {
                TreeNode *prev = curr->right;
                while (prev->left && prev->left != curr)
                    prev = prev->left;
                if (!prev->left)
                {
                    res.push_back(curr->val);
                    prev->left = curr;
                    curr = curr->right;
                }
                else
                {
                    // curr subtree traversed. revert the temp link and backtrack.
                    prev->left = nullptr;
                    curr = curr->left;
                }
            }
        }
        reverse(res.begin(), res.end());
        return res;
    }
} // namespace MorrisTraversal

namespace QuickSelect
{
    class kClosestPoint
    {
    public:
        vector<vector<int>> kClosest(vector<vector<int>> &points, int K)
        {
            int l = 0, r = points.size() - 1;
            while (true)
            {
                int p = partition(points, l, r);
                if (p == K - 1)
                {
                    break;
                }
                if (p < K - 1)
                {
                    l = p + 1;
                }
                else
                {
                    r = p - 1;
                }
            }
            return vector<vector<int>>(points.begin(), points.begin() + K);
        }

    private:
        bool farther(vector<int> &p, vector<int> &q)
        {
            return p[0] * p[0] + p[1] * p[1] > q[0] * q[0] + q[1] * q[1];
        }

        bool closer(vector<int> &p, vector<int> &q)
        {
            return p[0] * p[0] + p[1] * p[1] < q[0] * q[0] + q[1] * q[1];
        }

        int partition(vector<vector<int>> &points, int left, int right)
        {
            int pivot = left, l = left + 1, r = right;
            while (l <= r)
            {
                if (farther(points[l], points[pivot]) && closer(points[r], points[pivot]))
                {
                    swap(points[l++], points[r--]);
                }
                if (!farther(points[l], points[pivot]))
                {
                    l++;
                }
                if (!closer(points[r], points[pivot]))
                {
                    r--;
                }
            }
            swap(points[pivot], points[r]);
            return r;
        }
    };
} // namespace QuickSelect

namespace SegmentTree
{
    class SegmentTreeNode
    {
    public:
        SegmentTreeNode(int start, int end, int sum,
                        SegmentTreeNode *left = nullptr,
                        SegmentTreeNode *right = nullptr) : start(start),
                                                            end(end),
                                                            sum(sum),
                                                            left(left),
                                                            right(right) {}
        SegmentTreeNode(const SegmentTreeNode &) = delete;
        SegmentTreeNode &operator=(const SegmentTreeNode &) = delete;
        ~SegmentTreeNode()
        {
            delete left;
            delete right;
            left = right = nullptr;
        }

        int start;
        int end;
        int sum;
        SegmentTreeNode *left;
        SegmentTreeNode *right;
    };

    class NumArray
    {
    public:
        NumArray(vector<int> nums)
        {
            nums_.swap(nums);
            if (!nums_.empty())
                root_.reset(buildTree(0, nums_.size() - 1));
        }

        void update(int i, int val)
        {
            updateTree(root_.get(), i, val);
        }

        int sumRange(int i, int j)
        {
            return sumRange(root_.get(), i, j);
        }

    private:
        vector<int> nums_;
        std::unique_ptr<SegmentTreeNode> root_;

        SegmentTreeNode *buildTree(int start, int end)
        {
            if (start == end)
            {
                return new SegmentTreeNode(start, end, nums_[start]);
            }
            int mid = start + (end - start) / 2;
            auto left = buildTree(start, mid);
            auto right = buildTree(mid + 1, end);
            auto node = new SegmentTreeNode(start, end, left->sum + right->sum, left, right);
            return node;
        }

        void updateTree(SegmentTreeNode *root, int i, int val)
        {
            if (root->start == i && root->end == i)
            {
                root->sum = val;
                return;
            }
            int mid = root->start + (root->end - root->start) / 2;
            if (i <= mid)
            {
                updateTree(root->left, i, val);
            }
            else
            {
                updateTree(root->right, i, val);
            }
            root->sum = root->left->sum + root->right->sum;
        }

        int sumRange(SegmentTreeNode *root, int i, int j)
        {
            if (i == root->start && j == root->end)
            {
                return root->sum;
            }
            int mid = root->start + (root->end - root->start) / 2;
            if (j <= mid)
            {
                return sumRange(root->left, i, j);
            }
            else if (i > mid)
            {
                return sumRange(root->right, i, j);
            }
            else
            {
                return sumRange(root->left, i, mid) + sumRange(root->right, mid + 1, j);
            }
        }
    };

    class NumArrayVectorBased
    {
    public:
        // Check the constructor for the initialization of these variables.
        vector<int> seg; // Segment Tree to be stored in a vector.
        int n;           // Length of the segment tree vector.

        // Function to build the Segment Tree
        // This function will fill up the child values first
        // (left == right) will satisfy for the leaf values and they will be updated in segment tree
        // seg[pos]=seg[2*pos+1]+ seg[2*pos+2]; -> This will help populate all other intermediate nodes
        // as well as the root node with the "sum" of their child nodes.
        // Finally we have a segment tree which has all 'nodes' with sum values of their child.
        void buildTree(vector<int> &nums, int pos, int left, int right)
        {
            if (left == right)
            {
                seg[pos] = nums[left];
                return;
            }
            int mid = (left + right) / 2;
            buildTree(nums, 2 * pos + 1, left, mid);
            buildTree(nums, 2 * pos + 2, mid + 1, right);
            seg[pos] = seg[2 * pos + 1] + seg[2 * pos + 2];
        }

        // Function to update a node in the segment tree
        // When a node is updated, then the change in the node value has to be propagated to the root
        // left, right -> represents the range of the node of segment tree. (Ex: [0, n-1] -> root)
        // pos       -> represents "position" in the segment tree data structure (Ex: 0 -> root)
        // Using left, right and pos -> we have all the information on the segment tree
        // Node at 'pos' in segment tree will have children at 2pos+1(left) and 2pos+2(right)

        // If index is less than left or more than right, then it is out of bound
        //      for this node's range so we ignore it and return (This makes the algo O(logn))
        // If left==right==index, then we found the index,
        //      update the value of the segment tree node & return
        // Otherwise, we need to find the index and we do this by checking child nodes (2pos+1, 2pos+2)
        //      update the segment tree pos with the updated child values' sum.
        //      This would help propagate the updated value of the chid indexes to the parent (through recursion)
        void updateUtil(int pos, int left, int right, int index, int val)
        {
            // no overlap
            if (index < left || index > right)
                return;

            // total overlap
            if (left == right)
            {
                if (left == index)
                    seg[pos] = val;
                return;
            }

            // partial overlap
            int mid = (left + right) / 2;
            updateUtil(2 * pos + 1, left, mid, index, val);      // left child
            updateUtil(2 * pos + 2, mid + 1, right, index, val); // right child
            seg[pos] = seg[2 * pos + 1] + seg[2 * pos + 2];
        }

        // Function to get the sum from the range [qlow, qhigh]
        // low, high -> represents the range of the node of segment tree. (Ex: [0, n-1] -> root)
        // pos       -> represents "position" in the segment tree data structure (Ex: 0 -> root)
        // Using low, high and pos -> we have all the information on the segment tree
        // Node at 'pos' in segment tree will have children at 2pos+1(left) and 2pos+2(right)

        // While searching for the range, there will be three cases: (Ex: arr: [-1, 4, 2, 0])
        //  - Total Overlap:    Return the value. (Ex: qlow, qhigh: 0,3 and low, high: 1,2)
        //  - No Overlap:       Return 0. (Ex: qlow, qhigh: 0,1 and low, high: 2,3)
        //  - Partial Overlap:  Search for it in both the child nodes and their ranges.
        //                      (Ex: Searching for 1,2 and node range is 0,1)
        int rangeUtil(int qlow, int qhigh, int low, int high, int pos)
        {
            if (qlow <= low && qhigh >= high)
            { // total overlap
                return seg[pos];
            }
            if (qlow > high || qhigh < low)
            { // no overlap
                return 0;
            }
            // partial overlap
            int mid = low + (high - low) / 2;
            return (rangeUtil(qlow, qhigh, low, mid, 2 * pos + 1) + rangeUtil(qlow, qhigh, mid + 1, high, 2 * pos + 2));
        }

        // Constructor for initializing the variables.
        NumArrayVectorBased(vector<int> &nums)
        {
            if (nums.size() > 0)
            {
                n = nums.size();
                seg.resize(4 * n, 0);         // Maximum size of a segment tree for an array of size n is 4n
                buildTree(nums, 0, 0, n - 1); // Build the segment tree
                // Print Segment Tree
                // for(int i=0;i<4*n;i++)
                //     cout<<seg[i]<<" ";
                // cout<<endl;
            }
        }

        // Update the segment Tree recurively using updateUtil
        void update(int index, int val)
        {
            if (n == 0)
                return;
            updateUtil(0, 0, n - 1, index, val);
        }

        // Get the sum for a specific range for the segment Tree
        int sumRange(int left, int right)
        {
            if (n == 0)
                return 0;
            return rangeUtil(left, right, 0, n - 1, 0);
            // query from left to right while segment tree is now at 'root' (pos=0) and range(0, n-1)
        }
    };
} // namespace SegmentTree

namespace BinaryIndexedTree
{
    // lowbit, return the lowest 1 bit
    int lowbit(int n)
    {
        return n & -n; // equivalent return n & (~n + 1);
    }
    // given length n vector nums, support:
    //   - add(x, k), add k to the i-th number
    //   - query(x, y), return sum(nums[x:y+1])

    class SingleModifyRangeQuery
    {
        int n;
        // t maintains a prefix sum for nums using a binary indexed tree
        vector<int> t;
        SingleModifyRangeQuery() {}

        void init(vector<int> nums)
        {
            n = nums.size();
            t = vector<int>(n + 1, 0);
            t[0] = nums[0];
            for (int i = 1; i <= n; i++)
            {
                add(i, nums[i - 1]);
            }
        }
        void add(int x, int k)
        {
            for (; x <= n; x += lowbit(x))
                t[x] += k;
        }
        int ask(int x) const
        {
            int ans = 0;
            for (; x > 0; x -= lowbit(x))
                ans += t[x];
            return ans;
        }
        // return prefix sum at this index, e.g. ask(7) = nums[0] + ... + nums[6]
        int query(int l, int r) const
        {
            return ask(r) - ask(l - 1);
        }
    };
    class RangeModifySingleQuery
    {
        int n;
        // now t maintains a diff vector, e.g. t[i] = nums[i] - nums[i-1]
        vector<int> t;
        RangeModifySingleQuery() {}

        void init(vector<int> nums)
        {
            n = nums.size();
            t = vector<int>(n, 0);
            nums.insert(nums.begin(), 0);
            for (int i = 1; i <= n; i++)
            {
                add(i, nums[i] - nums[i - 1]);
            }
        }
        void add(int x, int k)
        {
            for (; x <= n; x += lowbit(x))
                t[x] += k;
        }
        // add d for each of nums[l], ..., nums[r]
        void addRange(int l, int r, int d)
        {
            add(l, d);
            add(r + 1, -d);
        }
        // return prefix sum at this index, e.g. ask(7) = t[0] + ... + t[6]
        int ask(int x) const
        {
            int ans = 0;
            for (; x > 0; x -= lowbit(x))
                ans += t[x];
            return ans;
        }
        // return nums[x]
        int query(int x) const
        {
            return t[x] + ask(x);
        }
    };

    class RangeModifyRangeQuery
    {
        int n;
        // b is a diff vector, e.g. b[i] = nums[i] - nums[i-1]
        // t1 maintains a prefix sum of b[i]
        // t2 maintains a prefix sum of i * b[i]
        vector<int> t1;
        vector<int> t2;
        RangeModifyRangeQuery() {}
        void init(vector<int> nums)
        {
            n = nums.size();
            t1 = vector<int>(n + 1, 0);
            t2 = vector<int>(n + 1, 0);
            nums.insert(nums.begin(), 0);
            for (int i = 1; i <= n; i++)
            {
                add1(i, nums[i] - nums[i - 1]);
                add2(i, i * nums[i] - (i - 1) * nums[i - 1]);
            }
        }
        void add1(int x, int k)
        {
            for (; x <= n; x += lowbit(x))
                t1[x] += k;
        }
        // return prefix sum at this index
        int ask1(int x) const
        {
            int ans = 0;
            for (; x > 0; x -= lowbit(x))
                ans += t1[x];
            return ans;
        }
        void add2(int x, int k)
        {
            for (; x <= n; x += lowbit(x))
                t2[x] += k;
        }
        // return prefix sum at this index
        int ask2(int x) const
        {
            int ans = 0;
            for (; x; x -= lowbit(x))
                ans += t2[x];
            return ans;
        }

        // add d for each of nums[l], ..., nums[r]
        void addRange(int l, int r, int d)
        {
            add1(l, d);
            add1(r + 1, -d);
            add2(l, l * d);
            add2(r + 1, -(r + 1) * d);
        }
        // return nums[l] + ... + nums[r]
        int query(int l, int r) const
        {
            int right = (r + 1) * ask1(r) - ask2(r);
            int left = l * ask1(l - 1) - ask2(l - 1);
            return right - left;
        }
    };
} // namespace BinaryIndexedTree

namespace DigitDP
{
    // great explanation here: https://codeforces.com/blog/entry/53960
    class DigitDP
    {

        /// How many numbers x are there in the range a to b, where the digit d occurs exactly k times in x?

        vector<int> num;
        int d, k;
        int DP[12][12][2];
        /// DP[p][c][f] = Number of valid numbers <= b from this state
        /// p = current position from left side (zero based)
        /// c = number of times we have placed the digit d so far
        /// f = the number we are building has already become smaller than b? [0 = no, 1 = yes]

        int call(int pos, int cnt, int f)
        {
            if (cnt > k)
                return 0;

            if (pos == num.size())
            {
                if (cnt == k)
                    return 1;
                return 0;
            }

            if (DP[pos][cnt][f] != -1)
                return DP[pos][cnt][f];
            int res = 0;

            int LMT;

            if (f == 0)
            {
                /// Digits we placed so far matches with the prefix of b
                /// So if we place any digit > num[pos] in the current position, then the number will become greater than b
                LMT = num[pos];
            }
            else
            {
                /// The number has already become smaller than b. We can place any digit now.
                LMT = 9;
            }

            /// Try to place all the valid digits such that the number doesn't exceed b
            for (int dgt = 0; dgt <= LMT; dgt++)
            {
                int nf = f;
                int ncnt = cnt;
                if (f == 0 && dgt < LMT)
                    nf = 1; /// The number is getting smaller at this position
                if (dgt == d)
                    ncnt++;
                if (ncnt <= k)
                    res += call(pos + 1, ncnt, nf);
            }

            return DP[pos][cnt][f] = res;
        }

        int solve(int b)
        {
            num.clear();
            while (b > 0)
            {
                num.push_back(b % 10);
                b /= 10;
            }
            reverse(num.begin(), num.end());
            /// Stored all the digits of b in num for simplicity

            memset(DP, -1, sizeof(DP));
            int res = call(0, 0, 0);
            return res;
        }
    };
} // namespace DigitDP

namespace ConvexHull
{
    class ConvexHull
    {
        // Implementation of Andrew's monotone chain 2D convex hull algorithm.
        // Asymptotic complexity: O(n log n).
        // Practical performance: 0.5-1.0 seconds for n=1000000 on a 1GHz machine.

        typedef double coord_t;  // coordinate type
        typedef double coord2_t; // must be big enough to hold 2*max(|coordinate|)^2

        struct Point
        {
            coord_t x, y;

            bool operator<(const Point &p) const
            {
                return x < p.x || (x == p.x && y < p.y);
            }
        };

        // 3D cross product of OA and OB vectors, (i.e z-component of their "2D" cross product, but remember that it is not defined in "2D").
        // Returns a positive value, if OAB makes a counter-clockwise turn,
        // negative for clockwise turn, and zero if the points are collinear.
        coord2_t cross(const Point &O, const Point &A, const Point &B)
        {
            return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
        }

        // Returns a list of points on the convex hull in counter-clockwise order.
        // Note: the last point in the returned list is the same as the first one.
        vector<Point> convex_hull(vector<Point> P)
        {
            size_t n = P.size(), k = 0;
            if (n <= 3)
                return P;
            vector<Point> H(2 * n);

            // Sort points lexicographically
            sort(P.begin(), P.end());

            // Build lower hull
            for (size_t i = 0; i < n; ++i)
            {
                while (k >= 2 && cross(H[k - 2], H[k - 1], P[i]) <= 0)
                    k--;
                H[k++] = P[i];
            }

            // Build upper hull
            for (size_t i = n - 1, t = k + 1; i > 0; --i)
            {
                while (k >= t && cross(H[k - 2], H[k - 1], P[i - 1]) <= 0)
                    k--;
                H[k++] = P[i - 1];
            }

            H.resize(k - 1);
            return H;
        }
    };
} // namespace ConvexHull

namespace Combinatorics
{

    long long fact(int n)
    {
        if (n <= 1)
            return 1;
        long long res = 1;
        for (int i = 2; i <= n; i++)
        {
            res *= i;
        }
        return res;
    }

    class NChooseK
    {
        long long compute(int n, int k)
        {
            if (n < k)
                return 0;

            long long res = 1;

            for (long long i = 1; i <= k; ++i, n--)
            {
                res = res * (long long)(n) / i;
            }
            return res;
        }
    };

    class NPermuteK
    {
        long long compute(int n, int k)
        {
            return fact(n) / fact(n - k);
        }
    };
} // namespace Combinatorics

namespace StringSplit
{
    class UseSTL
    {
        vector<string> splitByWhitespace(string s)
        {
            istringstream ss(s);
            string token;
            vector<string> tokens;
            while (ss >> token)
                tokens.push_back(token);
            return tokens;
        }

        vector<string> splitAnyDelimiter(string s, string delimiter)
        {
            size_t pos_start = 0, pos_end, delim_len = delimiter.length();
            string token;
            vector<string> res;

            while ((pos_end = s.find(delimiter, pos_start)) != string::npos)
            {
                token = s.substr(pos_start, pos_end - pos_start);
                pos_start = pos_end + delim_len;
                res.push_back(token);
            }

            res.push_back(s.substr(pos_start));
            return res;
        }

        vector<string> splitSingleDelimiter(string s, char delimiter)
        {
            istringstream ss(s);
            string token;
            vector<string> res;

            while (getline(ss, token, delimiter))
                res.push_back(token);

            return res;
        }
        vector<string> splitUseBoost(string s, string delimiter)
        {
            vector<string> tokens;
            boost::split(tokens, s, boost::is_any_of(delimiter));
            return tokens;
        }
    };
} // namespace StringSplit

int main()
{
    return 0;
}