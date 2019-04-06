
using Test

include("walk.jl")

@info """ Solving the following board, where S is the initial state, P represent pits, and G is the goal state:
  +-----+-----+
  | 1,S |  2  |
  +-----+-----+
  | 3,P | 4,G |
  +-----+-----+
"""
@testset "Solve 2 x 2 environment" begin
  R = transpose(
      hcat([[ -Inf, -10., -Inf,  -1.,   -1.],
            [ -Inf,  10.,  -1., -Inf,   -1.],
            [  -1., -Inf, -Inf,  10.,  -10.],
            [  -1., -Inf, -10., -Inf,   10.]]...))

  # Transition matrix
  τ = transpose(
    hcat([[-1,  3, -1,  2, 1],
          [-1,  4,  1, -1, 2],
          [ 1, -1, -1,  4, 3],
          [ 2, -1,  3, -1, 4]]...))

  # Matrix of feasible actions
  actions_pool = [
    [down, right, none],
    [down,  left, none],
    [  up, right, none],
    [  up,  left, none]
  ]

  goal_state_index = 4

  Q = compute_Q_values(0.8, 100, R, τ, actions_pool, goal_state_index)
  trajectory = compute_trajectory(Q, τ, 1, goal_state_index)

  @test argmax(Q[1,:]) == (right |> Int)
  @test argmax(Q[2,:]) == (down |> Int)
  @test all(Q[goal_state_index,:].==[0.0, 0.0, 0.0, 0.0, 0.0])
  @test trajectory == [1, 2, 4]
end

@info """ Solving the following board, where S is the initial state, P represent pits, and G is the goal state:
  +-----+-----+-----+
  |  1  |  2  |  3  |
  +-----+-----+-----+
  |  4  | 5,P | 6,G | 
  +-----+-----+-----+
  | 7,S | 8,P |  9  |   
  +-----+-----+-----+
"""

@testset "Solve 3 x 3 environment" begin
  # Reward matrix
  R = transpose(
    hcat([[-Inf,  -1., -Inf,  -1.,  -1.],
          [-Inf, -10.,  -1.,  -1.,  -1.],
          [-Inf,  10.,  -1., -Inf,  -1.],
          [ -1.,  -1., -Inf, -10.,  -1.],
          [ -1.,  -1.,  -1.,  10., -10.],
          [ -1.,  -1., -10., -Inf,  10.],
          [ -1., -Inf, -Inf, -10.,  -1.],
          [-10., -Inf,  -1.,  -1., -10.],
          [ 10., -Inf, -10., -Inf,  -1.]]...))

  τ = transpose(
    hcat([[-1,  4, -1,  2, 1],
          [-1,  5,  1,  3, 2],
          [-1,  6,  2, -1, 3],
          [ 1,  7, -1,  5, 4],
          [ 2,  8,  4,  6, 5],
          [ 3,  9,  5, -1, 6],
          [ 4, -1, -1,  8, 7],
          [ 5, -1,  7,  9, 8],
          [ 6, -1,  8, -1, 9]]...))

  # Matrix of feasible actions
  actions_pool = [
    [down, right, none],
    [down,  left, right,  none],
    [down,  left, none],
    [  up,  down, right,  none],
    [  up,  down,  left,  right, none],
    [  up,  down,  left,  none],
    [  up, right,  none],
    [  up,  left,  right, none],
    [  up,  left,  none]
  ]

  goal_state_index = 6
  start_state_index = 7

  Q = compute_Q_values(0.8, 500, R, τ, actions_pool, goal_state_index)
  trajectory = compute_trajectory(Q, τ, start_state_index, goal_state_index)

  @test trajectory == [7, 4, 1, 2, 3, 6]
end

