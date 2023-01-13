### A Pluto.jl notebook ###
# v0.19.19

using Markdown
using InteractiveUtils

# ╔═╡ cedd5966-296c-48c6-a377-a269e57b589e
import Pkg

# ╔═╡ 446517e1-f830-42c7-bdf9-0c4fa4bf039a
Pkg.add(url="https://github.com/piever/PersistentCohomology.jl.git")

# ╔═╡ bd3af3ca-f0e9-4e29-b4bb-4008874e0975
using JuMP, MosekTools, SCS, COSMO, COPT

# ╔═╡ 513b9a57-66e0-4bb4-9234-da50035aa23e
using Graphs, Combinatorics

# ╔═╡ 700de3ac-b635-40c3-add6-1f8516358868
using SparseArrays

# ╔═╡ 5117188a-bc9a-4d51-bb1c-1e1838e64f35
using PersistentCohomology

# ╔═╡ bf5105c8-9932-46de-8a1d-f6493606465e
include("graph_utils.jl")

# ╔═╡ 916b5b87-8cc6-4665-8d68-bfd2450eb15e
begin
	use_complement = false
	graph_name = "ivan-7-bad"
	family = "chordal"
	G = load_family_graph(graph_name, family, use_complement)
end

# ╔═╡ cdf145d2-a2fe-4bec-af04-0cb922256e18
c = vietorisrips(adjacency_matrix(G), 4)

# ╔═╡ 03bb15b9-4993-4a1e-b3bc-9cfe0ebc22dd
keys(c[3])[1]

# ╔═╡ ac51c2b2-574a-4236-b841-38bcf32a66f5
issubset(keys(c[3])[1], (1,3,4,6))

# ╔═╡ 55ff3ea8-7718-4ed8-bac0-6962de9aabd9
findall(map((x) -> x > 2, [1, 2, 3]))

# ╔═╡ 53227026-6d69-4bab-a088-4cf6706eab51
for i in sort([2,3,1], rev=true)
	println(i)
end

# ╔═╡ Cell order:
# ╠═bd3af3ca-f0e9-4e29-b4bb-4008874e0975
# ╠═513b9a57-66e0-4bb4-9234-da50035aa23e
# ╠═700de3ac-b635-40c3-add6-1f8516358868
# ╠═cedd5966-296c-48c6-a377-a269e57b589e
# ╠═446517e1-f830-42c7-bdf9-0c4fa4bf039a
# ╠═5117188a-bc9a-4d51-bb1c-1e1838e64f35
# ╠═bf5105c8-9932-46de-8a1d-f6493606465e
# ╠═916b5b87-8cc6-4665-8d68-bfd2450eb15e
# ╠═cdf145d2-a2fe-4bec-af04-0cb922256e18
# ╠═03bb15b9-4993-4a1e-b3bc-9cfe0ebc22dd
# ╠═ac51c2b2-574a-4236-b841-38bcf32a66f5
# ╠═55ff3ea8-7718-4ed8-bac0-6962de9aabd9
# ╠═53227026-6d69-4bab-a088-4cf6706eab51
