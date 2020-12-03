% Use Majority of votes to select output class of each test and traning
% object.
function maj = majority(data, L)
    for i = 1:L
    count_1 = sum(data(:,i)==1);
    count_2 = sum(data(:,i)==2);
        if count_1 > count_2
            maj(i) = 1;
        else
            maj(i) = 2;
        end
    end
end