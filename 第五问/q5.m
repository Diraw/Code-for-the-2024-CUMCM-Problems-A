clc,clear
% 参数设置
a = 1.7 / (2 * pi);
theta = linspace(0, 32* pi,260000); % 设定 theta 的范围

% 计算螺线方程
r = a * theta;

% 转换为笛卡尔坐标
x = r .* cos(theta);
y = r .* sin(theta);
% plot(x,y)

idx = find(abs(r - 4.5) < 0.0001); % 找到接近半径4.5的点
idx = idx(1);

panru_x = flip(x(idx:end));
panru_y = flip(y(idx:end));
% plot(panru_x,panru_y)
% hold on
panchu_x = -panru_x;
panchu_y = -panru_y;
% plot(panchu_x,panchu_y);

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
% p5 = plot(circle1_x, circle1_y, 'r');
% p6 = plot(circle2_x, circle2_y, 'b');
% 
% axis on
% 
% hold off
circle1_x = flip(circle1_x);
circle1_y = flip(circle1_y);
all_x = [panru_x (circle1_x)  circle2_x flip(panchu_x)];
all_y = [panru_y (circle1_y) circle2_y flip(panchu_y)];
p0 = plot(all_x,all_y);
axis on

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
figure
index = alpha>(pi/2);
alpha(index) = pi-alpha(index);
plot(l_x,(alpha),'DisplayName','alpha')
% plot((temp_idx+1:delta:length(all_y)),(alpha),'DisplayName','alpha')
hold on
index = beta>(pi/2);
beta(index) = pi-beta(index);
plot(l_x,(beta),'DisplayName','beta')
% plot((temp_idx+1:delta:length(all_y)),(beta),'DisplayName','beta')
hold off
xlabel('移动距离/m')
ylabel('alpha,beta角度')
legend 
bili = abs(cos(alpha)./cos(beta));
% plot(bili)
plot(l_x,bili)
ylim([0,2])
xlabel('移动距离/m')
ylabel('前后运动速度比例系数')
[m,n]=max(bili);
[o,p]=min(bili);
index_x = all_x(n+temp_idx-1);
index_y = all_y(n+temp_idx-1);
index_x_min = all_x(p+temp_idx-1);
index_y_min = all_y(p+temp_idx-1);
figure
plot(all_x,all_y)
hold on
p1 = plot(index_x,index_y,'*','DisplayName','比例系数最大处位置');
% plot(index_x_min,index_y_min,'*');
% plot(circle2_x(end),circle2_y(end),'.')
legend(p1)
legend("Position", [0.50466,0.13559,0.38295,0.061054])
axis on
hold off
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
figure
index = alpha>(pi/2);
alpha(index) = pi-alpha(index);
plot(l_x,(alpha),'DisplayName','alpha')
% plot((temp_idx+1:delta:length(all_y)),(alpha),'DisplayName','alpha')
hold on
index = beta>(pi/2);
beta(index) = pi-beta(index);
plot(l_x,(beta),'DisplayName','beta')
% plot((temp_idx+1:delta:length(all_y)),(beta),'DisplayName','beta')
hold off
xlabel('移动距离/m')
ylabel('alpha,beta角度')
legend 
bili = abs(cos(alpha)./cos(beta));
% plot(bili)
plot(l_x,bili)
ylim([0,2])
xlabel('移动距离/m')
ylabel('前后运动速度比例系数')
[m,n]=max(bili);
[o,p]=min(bili);
index_x = all_x(n+temp_idx-1);
index_y = all_y(n+temp_idx-1);
index_x_min = all_x(p+temp_idx-1);
index_y_min = all_y(p+temp_idx-1);
figure
plot(all_x,all_y)
hold on
p1 = plot(index_x,index_y,'*','DisplayName','比例系数最大处位置');
% plot(index_x_min,index_y_min,'*');
% plot(circle2_x(end),circle2_y(end),'.')
legend(p1)
legend("Position", [0.50466,0.13559,0.38295,0.061054])
axis on
hold off

k2 = bili;

v_max = 2/(max(k1));
disp('龙头的最大速度为：');
disp(v_max);
disp('米每秒');


