using Flux
using Flux: @epochs



function generate_data(num_samples)
  train_data = [rand(1:10, rand(2:7)) for i in 1:num_samples]
  train_labels = (v -> sum(v)).(train_data)

  test_data = 2 .* train_data
  test_labels = 2 .* train_labels

  (train_data, train_labels, test_data, test_labels)
end


num_samples = 1000
num_epochs = 50

# generate our test data with the data generation function from above
train_data, train_labels, test_data, test_labels = generate_data(num_samples)
simple_rnn = Flux.RNN(1, 1, (x -> x))

function eval_model(x)
  out = simple_rnn.(x)[end]
  Flux.reset!(simple_rnn)
  out
end

loss(x, y) = abs(sum((eval_model(x) .- y)))

ps = Flux.params(simple_rnn)

# use the ADAM optimizer. It's a pretty good one!
opt = Flux.ADAM()

println("Training loss before = ", sum(loss.(train_data, train_labels)))
println("Test loss before = ", sum(loss.(test_data, test_labels)))

# callback function during training
evalcb() = @show(sum(loss.(test_data, test_labels)))

@epochs num_epochs Flux.train!(loss, ps, zip(train_data, train_labels), opt, cb = Flux.throttle(evalcb, 1))

# after training, evaluate the loss
println("Test loss after = ", sum(loss.(test_data, test_labels)))



