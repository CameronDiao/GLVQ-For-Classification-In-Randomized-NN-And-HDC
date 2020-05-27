printf('lendo problema %s ...\n', problema);

n_entradas= 34; n_clases= 6; n_fich= 1; fich{1}= 'dermatology.data'; n_patrons(1)= 366;

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
	for j = 1:n_entradas
	  t = fscanf(f,'%c',1); %printf('t=<%c>\n', t);
	  if t ~= '?'
		fseek(f,-1,SEEK_CUR); t = fscanf(f, '%i',1); x(i_fich,i,j) = t;
	  else
		x(i_fich,i,j) = 0;
	  end
	  fscanf(f,'%c',1);  % le e descarta a coma
%  	  printf('%g ', x(i_fich,i,j))
	end	
	cl(i_fich,i) = fscanf(f,'%i',1) - 1;  	% lectura da clase
%  	printf('cl= %i\n', cl(i_fich,i))
%  	if i==2 exit end
  end
  fclose(f);
end
