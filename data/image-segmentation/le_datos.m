printf('lendo problema %s ...\n', problema);

n_entradas= 19; n_clases= 7; 
n_fich= 2; fich{1}= 'segmentation.data'; n_patrons(1)= 210; fich{2}= 'segmentation.test'; n_patrons(2)= 2100;

n_max= max(n_patrons);
x = zeros(n_fich, n_max, n_entradas); cl= zeros(n_fich, n_max);

n_patrons_total = sum(n_patrons); n_iter=0;
clase={'BRICKFACE', 'SKY', 'FOLIAGE', 'CEMENT', 'WINDOW', 'PATH', 'GRASS'};

for i_fich=1:n_fich
  f=fopen(fich{i_fich}, 'r');
  if -1==f
		error('erro en fopen abrindo %s\n', fich{i_fich});
  end
  for i=1:n_patrons(i_fich)
   	fprintf(2,'%5.1f%%\r', 100*n_iter++/n_patrons_total);
   	t=fscanf(f,'%s',1);
   	for j=1:n_clases
			if strcmp(t, clase{j})
				cl(i_fich,i)=j-1; break
			end
		end
		for j = 1:n_entradas
			x(i_fich,i,j) = fscanf(f,'%g',1);
		end
  end
  fclose(f);
end
