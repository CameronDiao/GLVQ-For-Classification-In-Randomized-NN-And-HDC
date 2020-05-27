printf('lendo problema %s ...\n', problema);

n_entradas= 9; n_clases= 2; n_fich= 1; fich{1}= 'breast-cancer.data'; n_patrons(1)= 286;

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
	t= fscanf(f,'%s',1);   % lectura de clase
	if strcmp(t, 'no-recurrence-events')
	  cl(i_fich,i)= 0;
	elseif strcmp(t, 'recurrence-events')
	  cl(i_fich,i)=1;
	else
	  error('clase %s descoñecida', t)
	end
	for j = 1:n_entradas
	  t = fscanf(f,'%s',1);
	  if t == '?'
		x(i_fich,i,j) = 0; continue
	  end
	  if j==1
		val={'10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99'};
	  elseif j==2
		val={'lt40', 'ge40', 'premeno'};
	  elseif j==3
		val={'0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59'};
	  elseif j==4
		val={'0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '18-20', '21-23', '24-26','27-29', '30-32', '33-35', '36-39'};
	  elseif j==5
		val={'yes', 'no'};
	  elseif j==6
		val={'1', '2', '3'};
	  elseif j==7
		val={'left', 'right'};
	  elseif j==8
		val={'left_up', 'left_low', 'right_up','right_low', 'central'};
	  elseif j==9
		val={'yes', 'no'};
	  end
	  n=length(val);
	  for k=1:n
		if strcmp(t, val{k})
		  x(i_fich,i,j)=k; break
		end
	  end
	end
  end
  fclose(f);
end
