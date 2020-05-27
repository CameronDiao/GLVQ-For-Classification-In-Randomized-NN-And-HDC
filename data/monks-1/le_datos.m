printf('lendo problema %s ...\n', problema);

n_entradas= 6; n_clases= 2; 
n_fich= 2; fich{1}= 'monks-1.train'; n_patrons(1)= 124; fich{2}= 'monks-1.test'; n_patrons(2)= 432;

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
	cl(i_fich,i) = fscanf(f,'%i',1);  	% lectura da clase
	for j = 1:n_entradas
	  x(i_fich,i,j) = fscanf(f,'%i',1);
	end
	fscanf(f,'%s',1);  % le e descarta o ID
  end
  fclose(f);
end
