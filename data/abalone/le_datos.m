% abalone
printf('lendo problema abalone ...\n')
n_entradas= 8; n_clases= 3; n_fich= 1; fich{1}= 'abalone.data'; n_patrons(1)= 4177;

x = zeros(1, n_patrons, n_entradas); cl= zeros(1, n_patrons);

f=fopen(fich, 'r');
if -1==f
  error('erro en fopen abrindo %s\n', fich);
end
for i=1:n_patrons
  fprintf(2,'%5.1f%%\r', 100*i/n_patrons(1));
  t = fscanf(f, '%c', 1);
  switch t
  case 'M'
	x(1,i,1)=-1;
  case 'F'
	x(1,i,1)=0;
  case 'I'
	x(1,i,1)=1;
  end
  for j=2:n_entradas
	fscanf(f,'%c',1); x(1,i,j) = fscanf(f,'%f', 1); 
  end
  fscanf(f,'%c',1); t = fscanf(f,'%i', 1);
  if t < 9
	cl(1,i)=0;
  elseif t < 11
	cl(1,i)=1;
  else
	cl(1,i)=2;
  end
  fscanf(f,'%c',1);
%    disp(x(1,i,:)); disp(cl(1,i))
end
fclose(f);