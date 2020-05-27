printf('lendo problema %s ...\n', problema);

n_entradas= 3; n_clases= 3; 
n_fich= 2; fich{1}= 'hayes-roth.data'; n_patrons(1)= 132; fich{2}= 'hayes-roth.test'; n_patrons(2)= 28;

n_max= max(n_patrons);
x = zeros(n_fich, n_max, n_entradas); cl= zeros(n_fich, n_max);

n_patrons_total = sum(n_patrons); n_iter=0;

for i_fich=1:n_fich
  f=fopen(fich{i_fich}, 'r');
  if -1==f
	error('erro en fopen abrindo %s\n', fich{i_fich});
  end
  for i=1:n_patrons(i_fich)
   	fprintf(2,'%5.1f%%\r', 100*n_iter++/n_patrons_total);
   	if i_fich==1
	  fscanf(f,'%i',2);  % le e descarta as columnas 1 e 2
	else
	  fscanf(f,'%i',1);
	end
	for j = 1:n_entradas
	  x(i_fich,i,j) = fscanf(f,'%i',1);
	end
	cl(i_fich,i)=fscanf(f,'%i',1)-1;
  end
  fclose(f);
end
