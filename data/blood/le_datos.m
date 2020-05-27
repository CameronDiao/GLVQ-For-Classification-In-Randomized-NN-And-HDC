printf('lendo problema %s ...\n', problema);

n_entradas= 5; n_clases= 2; n_fich= 1; fich{1}= 'transfusion.data'; n_patrons(1)= 748; %fich{2}= ' '; n_patrons(2)= 0;

n_max= max(n_patrons);
x = zeros(n_fich, n_max, n_entradas); cl= zeros(n_fich, n_max);

n_patrons_total = sum(n_patrons); n_iter=0;

for i_fich=1:n_fich
  f=fopen(fich{i_fich}, 'r');
  if -1==f
	error('erro en fopen abrindo %s\n', fich{i_fich});
  end
  while 1  % le e descarta a 1ª liña
	c=fscanf(f,'%c',1);
	if c==sprintf('\n'); break end
  end
  for i=1:n_patrons(i_fich)
   	fprintf(2,'%5.1f%%\r', 100*n_iter++/n_patrons_total);
	x(i_fich,i,1) = fscanf(f,'%i',1); fscanf(f,'%c',2);
	x(i_fich,i,2) = fscanf(f,'%i',1); fscanf(f,'%c',1);
	x(i_fich,i,3) = fscanf(f,'%i',1); fscanf(f,'%c',1);
	x(i_fich,i,4) = fscanf(f,'%i',1); fscanf(f,'%c',2);
	cl(i_fich,i) = fscanf(f,'%i',1); fscanf(f,'%c',1);  	% lectura da clase
  end
  fclose(f);
end
