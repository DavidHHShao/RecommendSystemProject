function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

A=X*Theta';

J=1/2*sum((A(R==1)-Y(R==1)).^2)+lambda/2*(sum(sum(Theta.^2))+sum(sum(X.^2)));


for i=1:num_movies
         idx = find(R(i, :)==1);
         Theta_tmp=Theta(idx,:);
         Y_tmp=Y(i,idx);
         X_grad(i,:)=(X(i,:)*Theta_tmp'-Y_tmp)*Theta_tmp+lambda*X(i,:);
end
             
for j=1:num_users
         idx = find(R(:, j)==1);
         X_tmp=X(idx,:);
         Y_tmp=Y(idx,j);
         Theta_grad(j,:)=(Theta(j,:)*X_tmp'-Y_tmp')*X_tmp+lambda*Theta(j,:);
             
end

grad = [X_grad(:); Theta_grad(:)];

end
