printf('lendo problema %s ...\n', problema);

n_entradas= 17; n_clases= 15; n_fich= 1; fich{1}= 'primary-tumor.data'; 
n_patrons(1)= 330;  % orixinalmente 339 patróns; logo de suprimir clases, quedan 330

n_max= max(n_patrons);
x = zeros(n_fich, n_max, n_entradas); cl= zeros(n_fich, n_max);
n_patrons_total = sum(n_patrons); n_iter=0;
% o programa selecciona_patrons suprime as clases [6,10,15,16,20,21] porque teñen moi poucos patróns

for i_fich=1:n_fich
  f=fopen(fich{i_fich}, 'r');
  if -1==f
	error('erro en fopen abrindo %s\n', fich{i_fich});
  end
  for i=1:n_patrons(i_fich)
  	fprintf(2,'%5.1f%%\r', 100*n_iter++/n_patrons_total);
	cl(i_fich,i) = fscanf(f,'%i',1) - 1;  	% lectura da clase
	for j = 1:n_entradas
	  t = fscanf(f,'%s',1);
	  if t == '?'
		x(i_fich,i,j) = 0;
	  else
		x(i_fich,i,j) = str2double(t);
	  end
	end	
  end
  fclose(f);
end
