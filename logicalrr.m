
%%
%exercise 2 logical Regression
% Data is based on student result on two exams 
% and whether the student was
% admitted or not
% task is to build binary classification
%that estimates college admission base on student exam results

function[cost grad]= logicalrr( x,y)
%%
% inpu data
x= importdata('C:\Users\ec09143\Desktop\JOBS\ex2x.dat');
y= importdata('C:\Users\ec09143\Desktop\JOBS\ex2y.dat');
m= length(y);
% add column of ones to x
x= [ones(m,1),x];
%%
% find rows and columns
[r c]=size(x);
% find the output with logic 1
% find the output with logic 0
pos= find(y==1);
neg= find(y==0);
% plot the admitted students for both exams
% plot the not admitted students for both exams
plot(x(pos,2),x(pos,3),'+');
hold on
plot(x(neg,2),x(neg,3),'o')
hold on
xlabel('exam1')
ylabel('exam2')
% places a legend on the graphs line respectively
legend('Admitted','not admitted');
hold off
% initialize the parameters
theta=zeros(1,c);
% calculate the sigmoid function

lr= 1.0./(1.0+exp(-x*theta'));

h= lr;
% calculate cost function from the error
error= sum(-y.*log(h)-(1-y).*log(1-h));
cost=(1/r)*error;
% find the gradient for all possible values up to m
for i=1:r
    grad = (1/r).*x(i,:)' *(h(i)-y(i));
    theta= x'*x\x'*y;
end
grad
cost
theta
%%
% predict the result of students with exam 1 and exam 2
predict= [ 1.0000   38.5000   76.0000]*theta'
predict1=[1.0000   33.5000   68.0000]*theta'
