clc,clear
syms theta

t1 = 4.5/1.7*2*pi;
t1 = t1.*ones(1,200);
t2 = linspace(0,4.5,200)./1.7.*2.*pi;
lll = linspace(0,4.5,200);
g = 1.7/2/pi*sqrt(1+theta^2);
for i = 1:200
    s(i) = double(int(g,[t1(i),t2(i)]));
    j(i) = pi*lll(i);
end

figure
plot(lll,2*abs(s))
hold on
plot(lll,j)
h = 2*abs(s)+j;
plot(lll,h)
legend('螺线','调头圆弧','调头曲线）')
hold off
ylabel('长度/m')
xlabel('实际调头圆空间半径/m')

% 参数设置
a = 1.7 / (2 * pi);
theta = linspace(0, 32* pi,260000); % 设定 theta 的范围

% 计算螺线方程
r = a * theta;

% 转换为笛卡尔坐标
x = r .* cos(theta);
y = r .* sin(theta);

idx = find(abs(r - 4.5) < 0.0001); % 找到接近半径4.5的点
idx = idx(1);

panru_x = flip(x(idx:end));
panru_y = flip(y(idx:end));
panchu_x = -panru_x;
panchu_y = -panru_y;

x_idx = x(idx);
y_idx = y(idx);

% 定义两个点 A 和 B
x1 = x_idx; y1 = y_idx;
x2 = -x_idx; y2 = -y_idx;
A = [x1, y1];
B = [x2, y2];

% 计算线段长度 L
L = sqrt((x2 - x1)^2 + (y2 - y1)^2);

% 计算半径 r1 和 r2
r1 = L / 3;  % 半径比例为 2:1
r2 = L / 6;

% 计算圆心 C1 和 C2
C1 = A + (2/6) * (B - A);
C2 = A + (5/6) * (B - A);

% 计算线段方向
direction = (B - A) / L;

% 计算垂直方向
perp_direction = [-direction(2), direction(1)];

% 绘制半圆弧
theta1 = linspace(-pi/2, pi/2, 30000); % 半圆1
theta2 = linspace(pi/2, 3*pi/2, 30000); % 半圆2

circle1_x = C1(1) + r1 * cos(theta1) * perp_direction(1) - r1 * sin(theta1) * perp_direction(2);
circle1_y = C1(2) + r1 * cos(theta1) * perp_direction(2) + r1 * sin(theta1) * perp_direction(1);

circle2_x = C2(1) + r2 * cos(theta2) * perp_direction(1) - r2 * sin(theta2) * perp_direction(2);
circle2_y = C2(2) + r2 * cos(theta2) * perp_direction(2) + r2 * sin(theta2) * perp_direction(1);

% % 绘制
figure
circle1_x = flip(circle1_x);
circle1_y = flip(circle1_y);
all_x = [panru_x (circle1_x)  circle2_x flip(panchu_x)];
all_y = [panru_y (circle1_y) circle2_y flip(panchu_y)];
p0 = plot(all_x,all_y);
axis on

%文件地址
filename = 'result4.xlsx';
sheet1 = '位置';
sheet2 = '速度';

fengge = [length(panru_x) length(panru_x)+length(circle1_y)...
    length(panru_x)+length(circle1_y)+length(circle2_y)];
delta = 1;%移动步长，每次移动多少个数据点
idx_first = 1;


l0 = 2.86;
l = 0;
while abs(l-l0)>0.005
    l = cal_l(all_x(idx_first),all_y(idx_first),all_x(1),all_y(1));
    idx_first = idx_first+1;

end
idx_first = 1000;
n_lim = (length(all_x)-idx_first)/delta;
temp_idx = idx_first;
for i = 1:n_lim-1
    j = 1;
    l = 0;
    while abs(l-l0)>0.05
        l = cal_l(all_x(idx_first),all_y(idx_first),all_x(idx_first-j),all_y(idx_first-j));
        j = j+1;

    end
    point_idx = idx_first-(j-1);
    alpha1 = cal_alpha1(all_x(idx_first),all_y(idx_first),all_x(point_idx),all_y(point_idx));
    % alpha2 = cal_alpha2(all_x(idx_first),all_y(idx_first),idx_first,fengge);
    % alpha3 = cal_alpha2(all_x(point_idx),all_y(point_idx),point_idx,fengge);
    alpha2 = cal_slope(all_x,all_y,idx_first);
    alpha3 = cal_slope(all_x,all_y,point_idx);
    alpha(i) = abs(alpha2-alpha1);
    beta(i) = abs(alpha3-alpha1);
    
    if i ~= 1
        l_x(i) = l_x(i-1)+cal_l(all_x(idx_first),all_y(idx_first),all_x(idx_first-1),all_y(idx_first-1));
    else
        l_x(i) = 0;
    end
    idx_first = idx_first+delta;

end
% figure
index = alpha>(pi/2);
alpha(index) = pi-alpha(index);
% plot(l_x,(alpha),'DisplayName','alpha')
% plot((temp_idx+1:delta:length(all_y)),(alpha),'DisplayName','alpha')
% hold on
index = beta>(pi/2);
beta(index) = pi-beta(index);
% plot(l_x,(beta),'DisplayName','beta')
% % plot((temp_idx+1:delta:length(all_y)),(beta),'DisplayName','beta')
% hold off
% xlabel('移动距离/m')
% ylabel('alpha,beta角度')
% legend 
bili = abs(cos(alpha)./cos(beta));
% figure
% plot(l_x,bili)
% ylim([0,2])
% xlabel('移动距离/m')
% ylabel('前后运动速度比例系数')
[m,n]=max(bili);
% [o,p]=min(bili);
index_x = all_x(n+temp_idx-1);
index_y = all_y(n+temp_idx-1);
% index_x_min = all_x(p+temp_idx-1);
% index_y_min = all_y(p+temp_idx-1);
% figure
% plot(all_x,all_y)
% hold on
% p1 = plot(index_x,index_y,'*','DisplayName','比例系数最大处位置');
% % plot(index_x_min,index_y_min,'*');
% % plot(circle2_x(end),circle2_y(end),'.')
% legend(p1)
% legend("Position", [0.50466,0.13559,0.38295,0.061054])
% axis on
% hold off

k1 = bili;

fengge = [length(panru_x) length(panru_x)+length(circle1_y)...
    length(panru_x)+length(circle1_y)+length(circle2_y)];
delta = 1;%移动步长，每次移动多少个数据点
idx_first = 1;


l0 = 1.75;
l = 0;
while abs(l-l0)>0.0005
    l = cal_l(all_x(idx_first),all_y(idx_first),all_x(1),all_y(1));
    idx_first = idx_first+1;

end
idx_first = 1000;
n_lim = (length(all_x)-idx_first)/delta;
temp_idx = idx_first;
for i = 1:n_lim-1
    j = 1;
    l = 0;
    while abs(l-l0)>0.05
        l = cal_l(all_x(idx_first),all_y(idx_first),all_x(idx_first-j),all_y(idx_first-j));
        j = j+1;

    end
    point_idx = idx_first-(j-1);
    alpha1 = cal_alpha1(all_x(idx_first),all_y(idx_first),all_x(point_idx),all_y(point_idx));
    % alpha2 = cal_alpha2(all_x(idx_first),all_y(idx_first),idx_first,fengge);
    % alpha3 = cal_alpha2(all_x(point_idx),all_y(point_idx),point_idx,fengge);
    alpha2 = cal_slope(all_x,all_y,idx_first);
    alpha3 = cal_slope(all_x,all_y,point_idx);
    alpha(i) = abs(alpha2-alpha1);
    beta(i) = abs(alpha3-alpha1);
    
    if i ~= 1
        l_x(i) = l_x(i-1)+cal_l(all_x(idx_first),all_y(idx_first),all_x(idx_first-1),all_y(idx_first-1));
    else
        l_x(i) = 0;
    end
    idx_first = idx_first+delta;

end
% figure
index = alpha>(pi/2);
alpha(index) = pi-alpha(index);
% plot(l_x,(alpha),'DisplayName','alpha')
% plot((temp_idx+1:delta:length(all_y)),(alpha),'DisplayName','alpha')
% hold on
index = beta>(pi/2);
beta(index) = pi-beta(index);
% plot(l_x,(beta),'DisplayName','beta')
% % plot((temp_idx+1:delta:length(all_y)),(beta),'DisplayName','beta')
% hold off
% xlabel('移动距离/m')
% ylabel('alpha,beta角度')
% legend 
bili = abs(cos(alpha)./cos(beta));
% figure
% plot(l_x,bili)
% ylim([0,2])
% xlabel('移动距离/m')
% ylabel('前后运动速度比例系数')
[m,n]=max(bili);
[o,p]=min(bili);
index_x = all_x(n+temp_idx-1);
index_y = all_y(n+temp_idx-1);
index_x_min = all_x(p+temp_idx-1);
index_y_min = all_y(p+temp_idx-1);
% figure
% plot(all_x,all_y)
% hold on
% p1 = plot(index_x,index_y,'*','DisplayName','比例系数最大处位置');
% % plot(index_x_min,index_y_min,'*');
% % plot(circle2_x(end),circle2_y(end),'.')
% legend(p1)
% legend("Position", [0.50466,0.13559,0.38295,0.061054])
% axis on
% hold off

k2 = bili;

%0时刻的
postion_x(1) = circle1_x(1);
postion_y(1) = circle1_y(1);
pos_x = postion_x(1);
pos_y = postion_y(1);
l = [2.56 1.75*ones(1,222)];
postion_index = find(all_x == postion_x(1));
postion_index = postion_index(1);
v(1) = 1;
for i = 1:223
    len = 0;
    j = 1;
    while abs(len-l(i))>0.005
        len = cal_l(all_x(postion_index),all_y(postion_index),all_x(postion_index-j),all_y(postion_index-j));
        j = j+1;
    end
    postion_x(i+1) = all_x(postion_index-j+1);
    postion_y(i+1) = all_y(postion_index-j+1);

    %更新postion_index
    postion_index = postion_index-j+1;

    if i == 1
        v(i+1) = v(i)*k1(postion_index-1000+1);
    else
        v(i+1) = v(i)*k2(postion_index-1000+1);
    end
end
% 写入数据到指定的单元格范围
result_po_1 = zeros(1, 2 * length(postion_x));
result_po_1(1:2:end) = postion_x;
result_po_1(2:2:end) = postion_y;
formatted_matrix_1 = arrayfun(@(x) sprintf('%.6f', x),result_po_1', 'UniformOutput', false);
writecell(formatted_matrix_1, filename, 'Sheet', sheet1, 'Range', 'CX2');

formatted_matrix_1 = arrayfun(@(x) sprintf('%.6f', x),v', 'UniformOutput', false);
writecell(formatted_matrix_1, filename, 'Sheet', sheet2, 'Range', 'CX2');


% plot(postion_x,postion_y,'-*')


%首先对第一个点
%套个for
%后退100秒的
postion_x(1) = circle1_x(1);
postion_y(1) = circle1_y(1);
postion_index_first = find(all_x == postion_x(1));
postion_index_first = postion_index_first(1);
l = [2.56 1.75*ones(1,222)];
result_po_2 = zeros(224*2,100);
result_v_2 = zeros(224,100);
for i_100 = 1:100 
    %首先要找到第一个点在哪
    
    len_first = 0;
    j = 1;
    while abs(len_first-1)>0.0025
        len_first = abs(l_x(postion_index_first)-l_x(postion_index_first-j));
        j = j+1;
    end
    postion_index_first = postion_index_first-j+1;

    postion_x(1) = all_x(postion_index_first);
    postion_y(1) = all_y(postion_index_first);
    pos_x = postion_x(1);
    pos_y = postion_y(1);
    
    postion_index = find(all_x == postion_x(1));
    postion_index = postion_index(1);
    for i = 1:223
        len = 0;
        j = 1;
        while abs(len-l(i))>0.008
            len = cal_l(all_x(postion_index),all_y(postion_index),all_x(postion_index-j),all_y(postion_index-j));
            j = j+1;
        end
        postion_x(i+1) = all_x(postion_index-j+1);
        postion_y(i+1) = all_y(postion_index-j+1);
    
        %更新postion_index
        postion_index = postion_index-j+1;

        if i == 1
            v(i+1) = v(i)*k1(postion_index-1000+1);
        else
            v(i+1) = v(i)*k2(postion_index-1000+1);
        end
    end
    result_po_temp_2 = zeros(1, 2 * length(postion_x));
    result_po_temp_2(1:2:end) = postion_x;
    result_po_temp_2(2:2:end) = postion_y;
    result_v_2(:,101-i_100) = v';
    result_po_2(:,101-i_100) = result_po_temp_2';
    % figure
    % plot(postion_x,postion_y,'-*');
end


formatted_matrix_2 = arrayfun(@(x) sprintf('%.6f', x),result_po_2, 'UniformOutput', false);
writecell(formatted_matrix_2, filename, 'Sheet', sheet1, 'Range', 'B2');

formatted_matrix_2 = arrayfun(@(x) sprintf('%.6f', x),result_v_2, 'UniformOutput', false);
writecell(formatted_matrix_2, filename, 'Sheet', sheet2, 'Range', 'B2');
% plot(postion_x,postion_y,'-*')


%首先对第一个点
%套个for
%前进100秒的
postion_x(1) = circle1_x(1);
postion_y(1) = circle1_y(1);
postion_index_first = find(all_x == postion_x(1));
postion_index_first = postion_index_first(1);
l = [2.56 1.75*ones(1,222)];
result_po_3 = zeros(224*2,100);
result_v_3 = zeros(224,100);
for i_100 = 1:100 
    %首先要找到第一个点在哪
    
    len_first = 0;
    j = 1;
    while abs(len_first-l(1))>0.0025
        len_first = abs(l_x(postion_index_first)-l_x(postion_index_first+j));
        j = j+1;
    end
    postion_index_first = postion_index_first+j-1;

    postion_x(1) = all_x(postion_index_first);
    postion_y(1) = all_y(postion_index_first);
    pos_x = postion_x(1);
    pos_y = postion_y(1);
    
    postion_index = find(all_x == postion_x(1));
    postion_index = postion_index(1);
    for i = 1:223
        len = 0;
        j = 1;
        while abs(len-l(i))>0.008
            len = cal_l(all_x(postion_index),all_y(postion_index),all_x(postion_index-j),all_y(postion_index-j));
            j = j+1;
        end
        postion_x(i+1) = all_x(postion_index-j+1);
        postion_y(i+1) = all_y(postion_index-j+1);
    
        %更新postion_index
        postion_index = postion_index-j+1;
        if i == 1
            v(i+1) = v(i)*k1(postion_index-1000+1);
        else
            v(i+1) = v(i)*k2(postion_index-1000+1);
        end
    end

    result_po_temp_3 = zeros(1, 2 * length(postion_x));
    result_po_temp_3(1:2:end) = postion_x;
    result_po_temp_3(2:2:end) = postion_y;
    result_v_3(:,i_100) = v';
    result_po_3(:,i_100) = result_po_temp_3';

    % figure
    % plot(postion_x,postion_y,'-*');
    
end
formatted_matrix_3 = arrayfun(@(x) sprintf('%.6f', x),result_po_3, 'UniformOutput', false);
writecell(formatted_matrix_3, filename, 'Sheet', sheet1, 'Range', 'CY2');

formatted_matrix_3 = arrayfun(@(x) sprintf('%.6f', x),result_v_3, 'UniformOutput', false);
writecell(formatted_matrix_3, filename, 'Sheet', sheet2, 'Range', 'CY2');
% plot(postion_x,postion_y,'-*');









