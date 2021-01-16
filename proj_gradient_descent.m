function [u, residuals] = proj_gradient_descent(grad_f, hess_f,x_u, u, gamma, maxiter)

L = max(eig(hess_f));
t = 1 / L;
residuals = zeros(maxiter, 1);

for k = 1:maxiter
    u_prec = u;
    x = x_u(u);
    %u = max(u - t * grad_f(u), 0);
    u = max(u - t * grad_f(x), 0);
    residuals(k) = norm(u - u_prec);
    if norm(u - u_prec) < gamma
        break
    end
end

residuals = residuals(1:k);

fprintf("Projected gradient descent for x >= 0\n");
fprintf("Minimum attained in %d iters\n\n", k);
