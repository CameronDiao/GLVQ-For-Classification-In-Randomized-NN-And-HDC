printf('lendo problema %s ...\n', problema);

n_entradas= 101; n_clases= 100; n_fich= 1; fich{1}= 'lrs.data'; n_patrons(1)= 531;

n_max= max(n_patrons);
x = zeros(n_fich, n_max, n_entradas); cl= zeros(n_fich, n_max);

n_patrons_total = sum(n_patrons); n_iter=0;

for i_fich=1:n_fich
  f=fopen(fich{i_fich}, 'r');
  if -1==f
	error('erro en fopen abrindo %s\n', fich{i_fich});
  end
  for i=1:n_patrons(i_fich)
%    	fprintf(2,'%5.1f%%\r', 100*n_iter++/n_patrons_total);
	fscanf(f,'%s',1);
	cl(i_fich,i) = fscanf(f,'%i',1);  	% lectura da clase
  	printf('cl= %i: ', cl(i_fich,i))
  	if cl(i_fich,i)>100 exit end
	for j = 1:n_entradas
	  x(i_fich,i,j) = fscanf(f,'%g',1);
  	  printf('%g ', x(i_fich,i,j))
	end
  	printf('\n')
%  	if i==2 exit end
  end
  fclose(f);
end
