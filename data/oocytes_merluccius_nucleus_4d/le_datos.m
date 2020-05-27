printf('lendo problema %s ...\n', problema);

n_entradas= 41; n_clases= 2; n_fich= 1; fich{1}= 'coocurrencia_1_2_4_8_gris_probability_gris_statistics_5_color_nucleos_R.dat'; n_patrons(1)= 1022;

n_max= max(n_patrons);
x = zeros(n_fich, n_max, n_entradas); cl= zeros(n_fich, n_max);
n_patrons_total = sum(n_patrons); n_iter=0;

for i_fich=1:n_fich
  f=fopen(fich{i_fich}, 'r');
  if -1==f
		error('erro en fopen abrindo %s\n', fich{i_fich});
  end
  for i=1:n_entradas
		fscanf(f,'%s',1);
	end
	fscanf(f,'%s',1);
  for i=1:n_patrons(i_fich)
  	fprintf(2,'%5.1f%%\r', 100*n_iter++/n_patrons_total);
		fscanf(f,'%i',1);
		for j = 1:n_entradas
			x(i_fich,i,j) = fscanf(f,'%g',1);
%  			printf('%g ', x(i_fich,i,j))
	  end
	  s = fscanf(f,'%s',1);    	% lectura da clase
		if strcmp(s, 'cn')
			cl(i_fich,i) = 0;
		elseif strcmp(s, 'sn')
			cl(i_fich,i) = 1;
		else
			error('clase %s descoñecida', s)
		end
%  		printf('cl= %i\n', cl(i_fich,i))
%  		if i==2 exit end
	end	
  fclose(f);
end
