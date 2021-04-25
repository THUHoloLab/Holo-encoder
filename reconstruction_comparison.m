clearvars;clc;close all

P1 = im2double(imread('./images/castle_GS.png'));
P2 = im2double(imread('./images/castle_Wirtinger.png'));
P3 = im2double(imread('./images/castle_HoloEncoder.png'));

[m,n] = size(P1);
holo1 = padarray(exp(1i*2*pi*P1),[m/2 n/2]);
holo2 = padarray(exp(1i*2*pi*P2),[m/2 n/2]);
holo3 = padarray(exp(1i*2*pi*P3),[m/2 n/2]);

lambda = 532e-6;    % wavelength
z = 160;            % propagation distance
dp = 0.00374;       % pixel pitch

Lx = dp*n;
Ly = dp*m;
f_max = 0.5 / dp;
du = 1 / Lx;
dv = 1 / Ly;
[u,v] = meshgrid(-f_max:du/2:f_max-du/2,-f_max:dv/2:f_max-dv/2);

I1 = asmprop(holo1,lambda,z,u,v);
figure,imshow(I1(1081:3240,1921:5760),[]);title('GS')
I2 = asmprop(holo2,lambda,z,u,v);
figure,imshow(I2(1081:3240,1921:5760),[]);title('Wirtinger')
I3 = asmprop(holo3,lambda,z,u,v);
figure,imshow(I3(1081:3240,1921:5760),[]);title('Holo-encoder')

function I = asmprop(holo,lambda,z,u,v)

FH = fftshift(ifft2(fftshift(holo)));
H = exp(1i*2*pi*z*sqrt(1/lambda^2 - u.^2 - v.^2));
U = fftshift(fft2(fftshift(FH.*H)));
I = abs(U).^2;

end