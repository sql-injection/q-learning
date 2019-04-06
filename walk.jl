@enum Action begin
  up = 1
  down = 2
  left = 3
  right = 4
  none = 5
end

function compute_Q_values(
  γ::Float64,
  num_episodes::Int,
  R::AbstractMatrix,
  τ::AbstractMatrix,
  actions_pool::Array{Array{Action, 1}, 1},
  goal_state_index::Int)

  # Q-value table
  num_states, num_actions = size(R)
  Q = zeros((num_states, num_actions))

  function q(s::Int, a::Int)
    r = R[s, a]
    s′ = τ[s, a]
    return r + γ * maximum([Q[s′, (a′ |> Int)] for a′ in actions_pool[s′]])
  end

  # Update q-values over episodes
  for i in 1:num_episodes
    s = rand(1:num_states)

    while s != goal_state_index
      actions = actions_pool[s]
      next_action = actions[rand(1:lastindex(actions))]
      next_action = next_action |> Int
      Q[s, next_action] = q(s, next_action)
      s = τ[s, next_action]
    end
  end

  return Q
end

function compute_trajectory(Q::AbstractMatrix, τ, start_state_index::Int, goal_state_index::Int)
  # Follow policy based on Q-values
  s = start_state_index
  trajectory = []

  while s != goal_state_index
    next_action = argmax(Q[s, :])
    push!(trajectory, s)
    s = τ[s, next_action]
  end
  push!(trajectory, s)

  return trajectory
end
