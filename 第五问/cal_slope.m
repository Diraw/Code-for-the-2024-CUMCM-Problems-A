function a = cal_slope(x,y,index)

    dx = x(index + 1) - x(index - 1);
    dy = y(index + 1) - y(index - 1);
    slope = dy / dx;
    a = atan(slope);
    if a<0
        a = a+pi;
    end
end