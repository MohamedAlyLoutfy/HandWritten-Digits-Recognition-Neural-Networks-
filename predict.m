function p = predict(Theta1, Theta2, X)

m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);


a1 = [ones(m, 1) X];

z2 = a1 * Theta1';
a2 = sigmoid(z2);

a2 = [ones(m, 1) a2];
z3 = a2 * Theta2';
hypo = sigmoid(z3);
[probability indices] = max(hypo');
p = indices';








end
