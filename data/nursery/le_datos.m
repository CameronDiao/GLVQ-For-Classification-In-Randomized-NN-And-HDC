printf('lendo problema %s ...\n', problema);

n_entradas= 8; n_clases= 5; n_fich= 1; fich{1}= 'nursery.data'; n_patrons(1)= 12960;

n_max= max(n_patrons);
x = zeros(n_fich, n_max, n_entradas); cl= zeros(n_fich, n_max);
n_patrons_total = sum(n_patrons); n_iter=0;
n_val_entrada = [3 5 4 4 3 2 3 3]; max_n_val_entrada=max(n_val_entrada);
val_entrada=cell(n_entradas, max_n_val_entrada);
clase={'not_recom', 'recommend', 'very_recom', 'priority', 'spec_prior'};

f=fopen('valores_entradas.dat', 'r');
if -1==f
  error('erro en fopen abrindo valores_entradas.dat')
end
for i=1:n_entradas
  for j=1:n_val_entrada(i)
	val_entrada{i,j} = fscanf(f,'%s', 1);
%    	printf('%s ', val_entrada{i,j})
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
	for j = 1:n_entradas
	  t = fscanf(f,'%s',1);
	  for k=1:n_val_entrada(j)
		if strcmp(t, val_entrada{j,k})
		  x(i_fich,i,j) = k; break
		end
	  end
	end	
	t = fscanf(f,'%s',1);  	% lectura da clase
	for k=1:n_clases
	  if strcmp(t, clase{k})
		cl(i_fich,i) = k-1; break
	  end
	end
  end
  fclose(f);
end
