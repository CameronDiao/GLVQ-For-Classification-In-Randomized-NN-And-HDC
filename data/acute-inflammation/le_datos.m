% acute
printf('lendo problema acute ...\n');
n_entradas= 6; n_clases= 2; n_fich= 1; fich{1}= 'diagnosis.data'; n_patrons(1)= 120;

x = zeros(1, n_patrons, n_entradas); cl= zeros(1, n_patrons);

f=fopen(fich, 'r');
if -1==f
  error('erro en fopen abrindo %s\n', fich);
end
for i=1:n_patrons
  fprintf(2,'%5.1f%%\r', 100*i/n_patrons(1));
  x(1,i,1) = fscanf(f, '%g', 1);
  for j=2:n_entradas
	t = fscanf(f,'%s', 1);
	if strcmp(t, 'no')
	  x(1,i,j)=-1;
	else
	  x(1,i,j)=1;
	end
  end
  t = fscanf(f,'%s', 1);
  if strcmp(t, 'no')
	cl(1,i)=0;
  else
	cl(1,i)=1;
  end
  fscanf(f,'%s',1);
end
fclose(f);