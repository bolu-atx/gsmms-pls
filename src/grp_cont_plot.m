grp1 = find(y1<1.1)
grp1 = grp1(grp1<1200)
grp1
grp1 = grp1(grp1<1100)
grp1 = grp1(grp1>1000)
grp1
grp2 = grp1
grp1 = 1016:1040
model
model.P
P = model.P
model
model.xpara1
Xs = pretreat(X_reduced)
Xs = pretreat(X_reduced);
Xs = pretreat(X_reduced,'autoscaling');
Xs
X_grp1 = Xs(grp1,:);
X_grp2 = Xs(grp2,:);
grp1
grp2
grp1
grp2
model
model.T(grp2,:)
T_grp2 = model.T(grp2,:);


%%
cont = nan(size(grp2),size(Xs,2));
for j = 1:size(Xs,2);
    for i = 1:size(T_grp2,1);
        cont(i,j) = T_grp2(i,:)*(P(j,:)'*(X_grp2(i,j)-mean(X_grp1(:,j))));
    end
end

cont(cont<0) = 0;
CONTv = sum(cont);