function [x, f_values] = gradient_descent(f, grad_f, hess_f, x, epsilon, maxiter, alpha, beta)

% I assume f(x) to be strongly convex
backtrack = false;
name = "fixed stepsize";
if exist('alpha','var') && exist('beta','var')
    backtrack = true;
    name = "backtracking";
end

L = max(eig(hess_f));
m = min(eig(hess_f));
t = 2 / (L + m);
max_grad_norm = sqrt(2*m*epsilon);
f_values = zeros(maxiter, 1);

%maxiter = int32(L / m * log10(1/epsilon));
%fprintf("maxiter = %d\n", maxiter);

if backtrack
    t_init = t;
end

for k = 1:maxiter
    f_values(k) = f(x);
    
    % If backtracking on t is needed
    if backtrack
        t = t_init;
        while f(x - t * grad_f(x)) > f(x) - alpha * t * norm(grad_f(x))^2
            t = beta * t;
        end
    end
    
    x = x - t * grad_f(x);
    if norm(grad_f(x)) <= max_grad_norm
        break
    end
end

fprintf("Gradient descent with %s\n", name);
fprintf("Minimum attained in %d iters\n\n", k);
f_values(k+1) = f(x);
f_values = f_values(1:k+1);
%x
