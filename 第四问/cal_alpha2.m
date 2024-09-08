function alpha2 = cal_alpha2(x,y,idx,fengge)
    C1 = [-0.940361202392951	-1.16489361410671];
    C2 = [1.88072240478590	2.32978722821341];
   
    if idx<=fengge(1)
        b = 1.7/(2*pi);
        r = cal_l(0,0,x,y);
        theta = r/b;
        tan_alpha2 = (sin(theta)+theta*cos(theta))/(cos(theta)-theta*sin(theta));
        alpha2 = atan(tan_alpha2);
    elseif idx>fengge(1)&&idx<=fengge(2)
        tan_alpha2 = -(x-C1(1))/(y-C1(2));
        alpha2 = atan(tan_alpha2);

    elseif idx>fengge(2)&&idx<=fengge(3)
        tan_alpha2 = -(x-C2(1))/(y-C2(2));
        alpha2 = atan(tan_alpha2);
    else
        b = -1.7/(2*pi);
        r = cal_l(0,0,x,y);
        theta = r/b;
        tan_alpha2 = (sin(theta)+theta*cos(theta))/(cos(theta)-theta*sin(theta));
        alpha2 = atan(tan_alpha2);
    end
    if alpha2<0
        alpha2 = alpha2+pi;
    end
end