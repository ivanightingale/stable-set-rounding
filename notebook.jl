### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 9a652b1a-0fbd-455b-a03a-889e169f3ebc
X = [1 0 0; 0 0 1; 1 0 1]

# ╔═╡ d918de66-794f-4c99-9569-54443eb6d676
X[3,:]

# ╔═╡ c7157aad-1fe6-493b-b6c3-8214be0d9329
filter(n -> X[end, n] > 0, collect(1:3))

# ╔═╡ Cell order:
# ╠═9a652b1a-0fbd-455b-a03a-889e169f3ebc
# ╠═d918de66-794f-4c99-9569-54443eb6d676
# ╠═c7157aad-1fe6-493b-b6c3-8214be0d9329
