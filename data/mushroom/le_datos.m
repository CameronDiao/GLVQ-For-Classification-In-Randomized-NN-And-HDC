printf('lendo problema %s ...\n', problema);

n_entradas= 22; n_clases= 2; n_fich= 1; fich{1}= 'agaricus-lepiota.data'; n_patrons(1)= 8124;

n_max= max(n_patrons);
x = zeros(n_fich, n_max, n_entradas); cl= zeros(n_fich, n_max);
n_patrons_total = sum(n_patrons); n_iter=0;
n_val_entrada = [6, 4, 10, 2, 9, 4, 3, 2, 12, 2, 7, 4, 4, 9, 9, 2, 4, 3, 8, 9, 6, 7]; max_n_val_entrada=max(n_val_entrada);
val_entrada=cell(n_entradas, max_n_val_entrada);

f=fopen('valores_entradas.dat', 'r');
if -1==f
  error('erro en fopen abrindo valores_entradas.dat')
end
for i=1:n_entradas
  for j=1:n_val_entrada(i)
	val_entrada{i,j} = fscanf(f,'%s', 1);
%  	printf('%s ', val_entrada{i,j})
  end
%    printf('\n')
end
fclose(f);

for i_fich=1:n_fich
  f=fopen(fich{i_fich}, 'r');
  if -1==f
	error('erro en fopen abrindo %s\n', fich{i_fich});
  end
  for i=1:n_patrons(i_fich)
  	fprintf(2,'%5.1f%%\r', 100*n_iter++/n_patrons_total);
	t= fscanf(f,'%s',1);
	if strcmp(t,'e')
	  cl(i_fich,i)=0;
	elseif strcmp(t,'p')
	  cl(i_fich,i)=1;
	else
	  error('clase %s descoñecida\n', t)
	end
	for j = 1:n_entradas
	  t = fscanf(f,'%s',1);
	  if t ~= '?'
		for k=1:n_val_entrada(j)
		  if strcmp(t, val_entrada{j,k})
			x(i_fich,i,j) = k; break
		  end
		end
	  else
		x(i_fich,i,j) = 0;
	  end
	end	
  end
  fclose(f);
end
