function class=bagging_ens(train,l_train,test)
    
    bag_tree=fitensemble(train,l_train,'bag',2,'Tree','type','classification');
    class=predict(bag_tree,test);
    
end