function alpha1 = cal_alpha1(x1,y1,x2,y2)
    alpha1 = atan((y2-y1)/(x2-x1));
    if alpha1<0
        alpha1 = alpha1+pi;
    end
end